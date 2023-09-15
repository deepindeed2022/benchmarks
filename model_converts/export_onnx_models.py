import os
import sys
import onnx
import onnxsim
from torch import nn
import torch
import torch.onnx
from torchvision import models as torch_models
import logging
import argparse
import onnxruntime as rt
import numpy as np
from model_utils import setup_logger, file_size


def convert_onnx(args, model_name, model_path, batch_size=8, image_size=224):
    if not model_name:
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        print("model_name = ", model_name)
    if args.fp16:
        output_fn = f'{model_name}_{image_size}_dynamic_bz{batch_size}_fp16_opset{args.opset_version}.onnx'
    else:
        output_fn = f'{model_name}_{image_size}_dynamic_bz{batch_size}_opset{args.opset_version}.onnx'

    dtype = torch.float16 if args.fp16 else torch.float32

    inputs = torch.rand(batch_size, 3, image_size,
                        image_size, dtype=dtype, device=0)
    input_names = ['input']
    output_names = ['output']
    unused_output = []
    origin_forward = None
    # Specify which model to use
    if model_name.startswith("efficientnet-b"):  # pytorch efficientnet
        from efficientnet_pytorch import EfficientNet
        model = EfficientNet.from_pretrained(model_name)
        model.set_swish(memory_efficient=False)
        model.cuda(0)
    elif model_name in ["resnet50", "resnet101"]:
        import timm
        model = timm.create_model(model_name, pretrained=True)
        model.cuda()
    elif model_name.startswith('swin'):
        import timm
        model = timm.create_model(model_name, pretrained=True)
        model.cuda()
    elif model_name.startswith("convnext"):
        model = eval(f"torch_models.{model_name}")(pretrained=True)
        model.cuda()
    elif model_name.startswith("vit"):
        model = eval(f"torch_models.{model_name}")(pretrained=True)
        model.cuda()
    elif model_name.startswith("resnext"):
        model = eval(f"torch_models.{model_name}")()
        model.cuda()
    elif model_name.startswith("yolox"):
        sys.path.append("./models/YOLOX")
        from yolox.exp import get_exp
        from yolox.models.network_blocks import SiLU
        from yolox.utils import replace_module
        exp = get_exp(f"models/YOLOX/exps/default/{model_name}.py", model_name)
        model = exp.get_model()
        ckpt = torch.load(model_path, map_location="cpu")

        model.eval()
        if "model" in ckpt:
            ckpt = ckpt["model"]
        model.load_state_dict(ckpt)
        model = replace_module(model, nn.SiLU, SiLU)
        model.head.decode_in_inference = False
        model.cuda()
    elif model_name == "yolov5s":
        unused_output = ["onnx::Sigmoid_353",
                         "onnx::Sigmoid_719", "onnx::Sigmoid_1085"]
        sys.path.append("models/yolov5")
        # model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        # start: load pretrained yolov5 model
        if os.path.exists(model_path):
            from models.experimental import attempt_load
            model = attempt_load(model_path, device=torch.device(
                "cpu"), inplace=True, fuse=True)
        # end:
        model.cuda()
    elif model_name.startswith("retinanet"):
        model = eval(f"torch_models.detection.{model_name}")()
        model.cuda()

    # test torch model inference performance

    from model_utils import test_infer_performance
    if origin_forward:
        model.forward = origin_forward
    if args.torchperf:
        for bz in [1, 4, 8, 16]:
            test_infer_performance(model=model, model_name=model_name, batch_size=bz, input_shape=(
                3, image_size, image_size), num_data=10240)

    if args.torchscript:
        model.eval()
        logging.info("start export torchscript model")
        output_fn = output_fn.replace(".onnx", ".torchscript")
        # sys.path.append("models/yolov5")
        # from models.yolo import Detect
        # for k, m in model.named_modules():
        #     if isinstance(m, Detect):
        #         m.inplace = False
        #         m.stride = [8, 16, 32, 64, 128]
        #         m.onnx_dynamic = False
        #         m.export = False
        # YOLOv5 TorchScript model export
        try:
            logging.debug(
                f'\n starting export with torch {torch.__version__}...')

            # d = {"shape": inputs.shape, "stride": int(max(model.stride)), "names": model.names}
            ts = torch.jit.trace(model, inputs, strict=False)
            extra_files = {'config.txt': ""}  # torch._C.ExtraFilesMap()

            ts.save(str(output_fn), _extra_files=extra_files)

            logging.info(
                f'export success, saved as {output_fn} ({file_size(output_fn):.1f} MB)')
            return
        except Exception as e:
            logging.info(f'export failure: {e}')
        return
    logging.info("start export onnx model")
    # export onnx format model
    torch.onnx.export(model, inputs, output_fn,
                      do_constant_folding=args.do_constant_folding,
                      opset_version=args.opset_version,
                      input_names=input_names,
                      output_names=output_names,
                      keep_initializers_as_inputs=True,
                      export_params=True,
                      dynamic_axes={
                          'input': {0: 'batch_size'},
                          'output': {0: 'batch_size'}
                      },
                      verbose=args.verbose)
    model_onnx = onnx.load(output_fn)  # load onnx model

    # remove initializer from onnx graph
    if model_onnx.ir_version < 4:
        print(
            'Model with ir_version below 4 requires to include initilizer in graph input'
        )
        return
    onnx_inputs = model_onnx.graph.input
    name_to_input = {}
    for input in onnx_inputs:
        name_to_input[input.name] = input

    for initializer in model_onnx.graph.initializer:
        if initializer.name in name_to_input:
            onnx_inputs.remove(name_to_input[initializer.name])
    onnx.checker.check_model(model_onnx)  # check onnx model
    # Simplify
    if args.simplify:
        try:
            logging.info(
                f'Simplify with onnx-simplifier {onnxsim.__version__}...')
            model_onnx, check = onnxsim.simplify(model_onnx, dynamic_input_shape=True, input_shapes={'input': [1, 3, image_size, image_size]},
                                                 unused_output=unused_output)
            assert check, 'assert simplification check failed'
            os.system(f"rm {output_fn}")
            output_fn = output_fn.replace(".onnx", ".sim.onnx")
            onnx.save(model_onnx, output_fn)
        except Exception as e:
            logging.exception(f'Simplifier failure: {e}')
    logging.info(
        f'Export success, saved as {output_fn} ({file_size(output_fn):.1f} MB)')

    if args.verify:
        logging.info("verify the onnx result")
        onnx_model = onnx.load(output_fn)
        onnx.checker.check_model(onnx_model)
        verify_inputs = torch.rand(
            1, 3, image_size, image_size, dtype=dtype, device=0)

        pytorch_result = model(verify_inputs)[0].cpu().detach().numpy()
        sess = rt.InferenceSession(output_fn, providers=[
                                   'CUDAExecutionProvider'])
        onnx_result = sess.run(
            None, {"input": verify_inputs.cpu().detach().numpy()})[0]

        diff = pytorch_result - onnx_result
        print("diff:\n{}\nmin:\n{}\nmax:\n{}".format(
            diff, diff.min(), diff.max()))
        if not np.allclose(pytorch_result, onnx_result):
            raise ValueError(
                'The outputs are different between Pytorch and ONNX')
        print('The outputs are same between Pytorch and ONNX')
    logging.info("export onnx and test finished")
    torch.save(model, f"{model_name}_{batch_size}_{image_size}.pth")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model_name', type=str,
                        default="efficient_b4_big_5cls", help='model name')
    parser.add_argument('--model_path', type=str,
                        default="models/dummy.model", help='model to run benchmark')
    parser.add_argument('-bs', '--batch_size', type=int,
                        default=8, help='batch size')
    parser.add_argument('--img_size', type=int, default=224, help='image size')
    parser.add_argument('--opset_version', type=int,
                        default=14, help='onnx opset version')
    parser.add_argument('--torchscript', action="store_true",
                        help="convert model as jit weight")
    parser.add_argument('--fp16', action="store_true",
                        help='use float16 or not')
    parser.add_argument('--simplify', action="store_true",
                        help='use simplifier')
    parser.add_argument('--do_constant_folding', action="store_true",
                        help='convert onnx, do constant value folding')
    parser.add_argument('--verify', action="store_true",
                        help='do verification')
    parser.add_argument('--torchperf', action="store_true",
                        help='do torch model performance test')
    parser.add_argument('--verbose', action="store_true",
                        help='print debug information or not')
    args = parser.parse_args()

    setup_logger(logname="run_onnx.log")
    convert_onnx(args, args.model_name, args.model_path,
                 batch_size=args.batch_size, image_size=args.img_size)
