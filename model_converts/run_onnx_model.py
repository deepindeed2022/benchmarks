import time
# from tkinter.font import names
import torch
import logging
import argparse
import numpy as np
import onnx
import onnxruntime
import onnxoptimizer
import onnxsim

from model_utils import setup_logger, file_size


def optimize_onnx_model(model_onnx, output_fn, use_simplify=True, use_optimizer=True, verbose=True):
    model_name_suffix = ""
    if use_optimizer:
        try:
            logging.info(f'Optimizer with onnxoptimizer ...')
            passnames = onnxoptimizer.get_available_passes()
            logging.info("Passnames: {}".format(passnames))
            model_onnx, check = onnxoptimizer.optimize(
                model_onnx, passes=['fuse_add_bias_into_conv', 'fuse_bn_into_conv'])
            assert check, 'assert onnxoptimizer check failed'
            output_fn = output_fn.replace(".onnx", ".opt.onnx")
            model_name_suffix += ".opt"
            onnx.save(model_onnx, output_fn)
        except Exception as e:
            logging.exception(f'OnnxOptimizer failure: {e}')
    if use_simplify:
        try:
            logging.info(
                f'Simplify with onnx-simplifier {onnxsim.__version__}...')
            model_onnx, check = onnxsim.simplify(
                model_onnx, dynamic_input_shape=False, input_shapes=None)
            assert check, 'assert simplification check failed'
            output_fn = output_fn.replace(".onnx", ".sim.onnx")
            model_name_suffix += ".sim"
        except Exception as e:
            logging.exception(f'OnnxSimplifier failure: {e}')
    onnx.save(model_onnx, output_fn)
    logging.info(
        f'Export success, saved as {output_fn} ({file_size(output_fn):.1f} MB)')
    return output_fn, model_name_suffix


def main(model_name, model_path, batch_size, image_size, num_data, model_fmt="onnx"):
    logging.info(f"startmodel {model_path} eval")
    use_fp16 = False
    if "fp16." in model_path:
        use_fp16 = True
    if model_fmt == "onnx":
        ort_performance(model_name, model_path, batch_size,
                        image_size, num_data, use_fp16=use_fp16)
    elif model_fmt == "torchscript":
        jit_performance(model_name, model_path, batch_size,
                        image_size, num_data, use_fp16=use_fp16)


def jit_performance(model_name, model_path, batch_size, image_size, num_data, use_fp16=True):
    extra_files = {'config.txt': ""}  # torch._C.ExtraFilesMap()
    model = torch.jit.load(model_path, _extra_files=extra_files).float()
    model.to("cuda:0")
    from run_pytorch import test_infer_performance
    logging.debug("test torchscript model performance")
    test_infer_performance(model, model_name, batch_size, num_data=num_data, input_shape=(
        3, image_size, image_size), use_fp16=use_fp16)


def ort_performance(model_name, model_path, batch_size, image_size, num_data, use_fp16=True):
    output_names = ['output']
    dtype = np.float16 if use_fp16 else np.float32
    inputs = np.random.rand(batch_size, 3, image_size,
                            image_size).astype(dtype)
    providers = [
        # ('TensorrtExecutionProvider', {
        #     'device_id': 0,
        #     'trt_max_workspace_size': 2147483648,
        #     'trt_fp16_enable': True,
        # }),
        # ('CUDAExecutionProvider', {
        #     'device_id': 0,
        #     'arena_extend_strategy': 'kNextPowerOfTwo',
        #     'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
        #     'cudnn_conv_algo_search': 'EXHAUSTIVE',
        #     'do_copy_in_default_stream': True,
        # }),
        "CUDAExecutionProvider"
    ]
    # set log level could ignore some useless warning log
    options = onnxruntime.SessionOptions()
    options.log_severity_level = 3
    model = onnxruntime.InferenceSession(
        model_path, providers=providers, sess_options=options)

    ort_inputs = {"input": inputs}
    ort_outputs = [out.name for out in model.get_outputs()]
    logging.info(f'test warm up onnx model')
    for _ in range(10):
        results = model.run(ort_outputs, ort_inputs)[0]
    torch.cuda.synchronize()
    logging.info(f'start testing onnx model')
    start_t = time.time()
    for _ in range(num_data // batch_size):
        results = model.run(ort_outputs, ort_inputs)[0]
    torch.cuda.synchronize()
    elapsed_time = time.time() - start_t
    logging.info('{} batch_size={} time: {:.4f} ms / image'.format(model_name,
                 batch_size, elapsed_time / num_data * 1000))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_name', type=str,
                        required=True, help='model to run')
    parser.add_argument('--model_path', type=str,
                        required=True, help='model path to run')
    parser.add_argument('-bs', '--batch_size', type=int,
                        default=8, help='batch size')
    parser.add_argument('--image_size', type=int,
                        default=224, help='image w/h size')
    parser.add_argument('-nd', '--num_data', type=int,
                        default=10240, help='num of data')
    parser.add_argument('--format', type=str,
                        default="onnx", help='num of data')
    args = parser.parse_args()
    setup_logger(logname="run_onnx.log")
    print("- TEST Model: ", args.model_path)
    main(args.model_name, args.model_path, args.batch_size,
         args.image_size, args.num_data, args.format)
