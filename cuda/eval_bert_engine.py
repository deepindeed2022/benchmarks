import time
import argparse
import logging
import numpy as np

import torch
import torch.cuda.nvtx as nvtx
import pycuda.driver as cuda
import torch.cuda.profiler as profiler
import tensorrt as trt

from utils import setup_logger


class HostDeviceMem:
    def __init__(self, dtype, size):
        self._host_mem = cuda.pagelocked_empty(size, dtype)
        self._device_mem = cuda.mem_alloc(self._host_mem.nbytes)

    @property
    def host(self):
        return self._host_mem

    @host.setter
    def host(self, val):
        self._host_mem = val

    @property
    def device(self):
        return self._device_mem

    @property
    def binding(self):
        return int(self._device_mem)

    def __str__(self):
        return "Host:\n" + str(self._host) + "\nDevice:\n" + str(self._device)

    def __repr__(self):
        return self.__str__()


class Buffer:
    def __init__(self, engine, config=None):
        self._name2shape = config
        self._inputs = {}
        self._outputs = {}
        self._bindings = []
        self._allocate_mem(engine)

    def _allocate_mem(self, engine):
        for name in engine:

            size = trt.volume(self._name2shape[name])
            # shape = engine.get_binding_shape(name)
            dtype = trt.nptype(engine.get_binding_dtype(name))
            hdm = HostDeviceMem(dtype, size)
            self._bindings.append(hdm.binding)
            if engine.binding_is_input(name=name):
                self._inputs[name] = hdm
            else:
                self._outputs[name] = hdm

    @property
    def inputs(self):
        return self._inputs

    @property
    def outputs(self):
        return self._outputs

    @property
    def bindings(self):
        return self._bindings

    def set_input(self, name, tensor):
        try:
            self._inputs[name].host = np.ascontiguousarray(tensor)
        except Exception as exc:
            print(exc)
            logging.warn(exc)


class TRTPredictor:
    def __init__(self, model_path, input_shape, output_shape, batch_size):
        # load plugin
        cuda.init()
        self._cuda_device = cuda.Device(0)
        self._cuda_ctx = self._cuda_device.make_context()
        self._engine = TRTPredictor.load_engine(model_path)

        self._context = self._engine.create_execution_context()

        if isinstance(input_shape, (tuple, list)):
            input_buffer_shape = {
                "input_ids": tuple(input_shape),
                "attention_mask": tuple(input_shape)
            }
        elif isinstance(input_shape, dict):
            input_buffer_shape = input_shape

        if isinstance(output_shape, (list, tuple)):
            output_buffer_shape = {
                "output": tuple(output_shape)
            }
        elif isinstance(output_shape, dict):
            output_buffer_shape = output_shape

        buffer_shape = input_buffer_shape
        for k, v in output_buffer_shape.items():
            buffer_shape[k] = v
        # buffer_shape.update(output_buffer_shape)
        self._buffer = Buffer(self._engine, buffer_shape)
        self._stream = cuda.Stream()

    def __del__(self):
        self._cuda_ctx.pop()

    @staticmethod
    def load_engine(model_path):
        TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
        plan = open(model_path, "rb").read()
        runtime = trt.Runtime(TRT_LOGGER)
        return runtime.deserialize_cuda_engine(plan)

    def set_input(self, inputs_dict):
        for name, tensor in inputs_dict.items():
            idx = self._engine.get_binding_index(name)
            self._context.set_binding_shape(idx, tensor.shape)
            self._buffer.set_input(name, tensor)

    def do_inference_v2(self):
        for name, inp in self._buffer.inputs.items():
            cuda.memcpy_htod_async(inp.device, inp.host, self._stream)

        self._context.execute_async_v2(
            bindings=self._buffer.bindings,
            stream_handle=self._stream.handle)

        for name, out in self._buffer.outputs.items():
            cuda.memcpy_dtoh_async(out.host, out.device, self._stream)
        self._stream.synchronize()

    def get_output(self):
        outputs_dict = {}
        for name, out in self._buffer.outputs.items():
            # idx = self._engine.get_binding_index(name)
            # shape = self._context.get_binding_shape(idx)
            # print(idx, name, shape)
            shape = self._context.get_tensor_shape(name)
            num = trt.volume(shape)
            outputs_dict[name] = np.copy(out.host[:num].reshape(shape))
        return outputs_dict

    def __call__(self, inputs_dict: dict):
        self._cuda_ctx.push()
        self.set_input(inputs_dict)
        self.do_inference_v2()
        outputs_dict = self.get_output()
        self._cuda_ctx.pop()
        return outputs_dict


def run_engine(args):
    num_data = 512
    # bert
    input_shape = {"input_ids": [args.batch_size, args.seq_length], "attention_mask": [
        args.batch_size, args.seq_length]}
    output_shape = {"output": [args.batch_size, 1000]}
    trtpredictor = TRTPredictor(
        args.engine, input_shape, output_shape, args.batch_size)
    inputs = {}
    for key, shape in input_shape.items():
        inputs[key] = np.random.rand(*shape).astype(np.int32)

    logging.info('warm up')
    for i in range(10):
        outputs_dict = trtpredictor(inputs_dict=inputs)
    torch.cuda.synchronize()
    logging.info('start testing trt engine')
    start_t = time.time()
    profiler.cudart().cudaProfilerStart()
    outputs_dict = trtpredictor(inputs_dict=inputs)
    profiler.cudart().cudaProfilerStop()
    torch.cuda.synchronize()
    elapsed_time = time.time() - start_t
    logging.info('{} batch_size={} time: {:.4f} ms / image'.format(args.engine,
                 args.batch_size, elapsed_time / num_data * 1000))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('engine', type=str,
                        help='Path to the optimized TensorRT engine')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--seq_length', type=int, default=256)
    args = parser.parse_args()
    setup_logger(logname="run_onnx.log")
    run_engine(args)
