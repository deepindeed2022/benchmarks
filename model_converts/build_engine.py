import logging

import tensorrt as trt

from model_utils import setup_logger

EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)


def build_engine(onnx_path, model_name, max_batch_size, image_size, percision="fp32", dla_core = None, gpu_fallback=None, verbose=False):
    TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.INFO)

    builder = trt.Builder(TRT_LOGGER)
    builder.max_batch_size = max_batch_size
    trt_version = [int(n) for n in trt.__version__.split('.')]
    logging.info(f"Onnx File: {onnx_path}, max_batch_size={max_batch_size}, image_size={image_size}")
    logging.info(f'Starting export with TensorRT {trt.__version__}...')
    network = builder.create_network(EXPLICIT_BATCH)
    parser = trt.OnnxParser(network, TRT_LOGGER)
    if not parser.parse_from_file(onnx_path):
        logging.error('ERROR: Failed to parse the ONNX file.')
        for error in range(parser.num_errors):
            logging.error(parser.get_error(error))
        return None
    # output trt input and output tensor name, we could ignore the output
    inputs = [network.get_input(i) for i in range(network.num_inputs)]
    outputs = [network.get_output(i) for i in range(network.num_outputs)]
    # for layer in network.layers:
    #     print(layer.name)
    logging.info(f'Network Description:')
    for inp in inputs:
        logging.info(f'input "{inp.name}" with shape {inp.shape} and dtype {inp.dtype}')
    for out in outputs:
        logging.info(f'output "{out.name}" with shape {out.shape} and dtype {out.dtype}')

    config = builder.create_builder_config()
    # set workspace
    if trt_version[0] >= 8:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, ((1 << 30)*15))
    else:
        config.max_workspace_size = ((1 << 30)*15)
    
    # tactic source
    if trt_version[0] >= 8:
        tactic_source = 1 << int(trt.TacticSource.CUBLAS) | 1 << int(trt.TacticSource.CUBLAS_LT)
        config.set_tactic_sources(tactic_source)

    profile = builder.create_optimization_profile()
    opt_bz = max_batch_size
    profile.set_shape("input", (1, 3, image_size, image_size), (opt_bz, 3, image_size, image_size), (max_batch_size, 3, image_size, image_size))
    config.add_optimization_profile(profile)


    if dla_core:
        logging.info("Use DLA Core to profile")
        config.default_device_type = trt.DeviceType.DLA
        config.DLA_core = dla_core
        if gpu_fallback:
            config.set_flag(trt.BuilderFlag.GPU_FALLBACK)
            
    config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
    if percision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)

    plan = builder.build_serialized_network(network, config)
    engine_path = "./{}_{}_bz{}_trt_{}_v{}.engine".format(model_name, image_size, max_batch_size, percision, trt.__version__)
    with open(engine_path, "wb") as f:
        f.write(plan)
        logging.info(f'Export success, saved as {engine_path}')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="models/model_best.pth.tar", help='model to run benchmark')
    parser.add_argument('--model_name', type=str, default="yolox-m", help='model name')
    parser.add_argument('-bz', '--max_batch_size', type=int, default=8, help='batch size')
    parser.add_argument('-imgsize', '--image_size', type=int, default=640, help='input image size')
    parser.add_argument('--fp16', action="store_true", help='use float16 or not')
    parser.add_argument('--verbose', action="store_true", help='print debug information or not')
    parser.add_argument('--dla_core', type=int, default=-1, help='use dla core')
    parser.add_argument('--gpu_fallback', action="store_true", help='gpu fallback')
    args = parser.parse_args()
    setup_logger(logname="run_onnx.log")
    build_engine(args.model_path, args.model_name, args.max_batch_size, args.image_size, percision="fp16" if args.fp16 else "fp32", dla_core=args.dla_core, gpu_fallback=args.gpu_fallback, verbose=args.verbose)
