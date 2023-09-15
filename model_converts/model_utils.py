from torch import nn
import time
from torch.cuda import amp
import torch
import logging
from pathlib import Path
import pkg_resources as pkg

model_locations = dict({
    "efficient_b4_big_5cls": "models/mpi/review-qc/human-image/efficient_b4_big_5cls.pth",
    "yolox-l-3w": "models/mpi/review-qc/human-image/yolox-l-3w-0617.pth",
    "yolox-l-4w": "models/mpi/review-qc/human-image/yolox-l-4w-0529.pth",
    "yolov5s": "models/yolov5s.pt",
    "yolov7x": "models/yolov7x.pt",
})


def is_parallel(model):
    # Returns True if model is of type DP or DDP
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)


def de_parallel(model):
    # De-parallelize a model: returns single-GPU model if model is of type DP or DDP
    return model.module if is_parallel(model) else model


def setup_logger(logname=None, verbose=False):
    FORMAT = '[%(asctime)s] p%(process)d {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s'
    logFormatter = logging.Formatter(FORMAT, datefmt='%m-%d %H:%M:%S')
    rootLogger = logging.getLogger()
    if verbose:
        rootLogger.setLevel(logging.DEBUG)
    else:
        rootLogger.setLevel(logging.INFO)

    if logname is not None:
        fileHandler = logging.FileHandler(logname)
        fileHandler.setFormatter(logFormatter)
        rootLogger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)


def file_size(path):
    # Return file/dir size (MB)
    path = Path(path)
    if path.is_file():
        return path.stat().st_size / 1E6
    elif path.is_dir():
        return sum(f.stat().st_size for f in path.glob('**/*') if f.is_file()) / 1E6
    else:
        return 0.0


def check_version(current='0.0.0', minimum='0.0.0', name='version ', pinned=False, hard=False, verbose=False):
    # Check version vs. required version
    current, minimum = (pkg.parse_version(x) for x in (current, minimum))
    result = (current == minimum) if pinned else (current >= minimum)  # bool
    # string
    s = f'{name}{minimum} required by YOLOv5, but {name}{current} is currently installed'
    if hard:
        assert result, s  # assert min requirements met
    if verbose and not result:
        logging.warning(s)
    return result


def test_infer_performance(model, model_name, batch_size, num_data, input_shape=None, use_fp16=False, use_mixed_precision=True):
    model.eval()
    model.cuda(0)
    print(f'warm up')
    # ``fuser0`` - enables only legacy fuser
    # ``fuser1`` - enables only NNC
    # ``fuser2`` - enables only nvFuser
    TORCH_NVFUSER_LABEL = "fuser1"
    dtype = torch.float16 if use_fp16 else torch.float32

    B, C, H, W = batch_size, 3, 224, 224
    if input_shape != None:
        assert len(
            input_shape) >= 3, "input shape size should >= 3, at least include (C, H, W) or (N, C, H, W)"
        B, C, H, W = batch_size, input_shape[-3], input_shape[-2], input_shape[-1]
    logging.info(f"{model_name} inference with ({B}, {C}, {H}, {W})")
    logging.info(f"warm up with {TORCH_NVFUSER_LABEL}")
    if check_version(torch.__version__, "1.12.0"):
        logging.info(f"use {torch.__version__}")
        for _ in range(10):
            data = torch.rand(B, C, H, W, dtype=dtype, device=0)
            with torch.jit.fuser(TORCH_NVFUSER_LABEL), torch.no_grad():
                o = model(data)
    else:
        for _ in range(10):
            data = torch.rand(B, C, H, W, dtype=dtype, device=0)
            with torch.no_grad():
                ret = model(data)

    torch.cuda.synchronize()
    print(f'start testing')
    logging.info(f'start testing')
    elapsed_time = 0

    for idx in range(num_data // batch_size):
        data = torch.rand(B, C, H, W, dtype=dtype, device=0)
        if check_version(torch.__version__, "1.12.0"):
            start_t = time.time()
            with torch.no_grad(), torch.jit.fuser(TORCH_NVFUSER_LABEL):
                ret = model(data)
            torch.cuda.synchronize()
            elapsed_time += time.time() - start_t
        else:
            start_t = time.time()
            with torch.no_grad(), amp.autocast(enabled=use_mixed_precision):
                ret = model(data)

            torch.cuda.synchronize()
            elapsed_time += time.time() - start_t

    print('[{}_{}x{}] time: {:.4f} ms / image'.format(model_name,
          W, H, elapsed_time / num_data * 1000))
    logging.info('[{}_{}x{}] time: {:.4f} ms / image'.format(model_name,
                 W, H, elapsed_time / num_data * 1000))


def test_bert_infer_performance(model, tokenizer, model_name, batch_size, num_data, max_length=128, use_mixed_precision=True):
    model.eval()
    model.cuda(0)

    print(f'warm up')
    for _ in range(10):
        inputs = tokenizer(["Welcome to My github!"] * batch_size,
                           return_tensors="pt", padding="max_length", max_length=max_length)

        for k in inputs:
            inputs[k] = inputs[k].cuda(0)
        with torch.no_grad(), amp.autocast(enabled=use_mixed_precision):
            outputs = model(**inputs)

    torch.cuda.synchronize()
    print(f'start testing')
    elapsed_time = 0
    for idx in range(num_data // batch_size):
        inputs = tokenizer(["Welcome to My github!"] * batch_size,
                           return_tensors="pt", padding="max_length", max_length=max_length)
        for k in inputs:
            inputs[k] = inputs[k].cuda(0)
        start_t = time.time()
        with torch.no_grad(), amp.autocast(enabled=use_mixed_precision):
            outputs = model(**inputs)
        torch.cuda.synchronize()
        elapsed_time += time.time() - start_t
    print('[{}_{}] time: {:.4f} ms / image'.format(model_name,
          max_length, elapsed_time / num_data * 1000))
