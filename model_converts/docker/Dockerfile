FROM nvcr.io/nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

ENV TRT_VERSION 8.6.1.6
ENV DEBIAN_FRONTEND=noninteractive

# Install requried libraries
RUN apt-get update && apt-get install -y --no-install-recommends    \
    libcurl4-openssl-dev                                            \
    wget                                                            \
    git                                                             \
    pkg-config                                                      \
    ssh                                                             \
    libssl-dev                                                      \
    build-essential

# Install python3
RUN apt-get install -y --no-install-recommends  \
      python3                                   \
      python3-pip                               \
      python3-dev                               \
      python3-wheel &&                          \
    cd /usr/local/bin &&                        \
    ln -s /usr/bin/python3 python &&            \
    ln -s /usr/bin/pip3 pip;

RUN v="${TRT_VERSION}-1+cuda11.8" &&                                                                                    \
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub && \
    apt-get update &&                                                                                                   \
    apt-get -y install  libnvinfer8=${v}                                                                                \
                        libnvonnxparsers8=${v}                                                                          \
                        libnvparsers8=${v}                                                                              \
                        libnvinfer-plugin8=${v}                                                                         \
                        libnvinfer-dev=${v}                                                                             \
                        libnvonnxparsers-dev=${v}                                                                       \
                        libnvparsers-dev=${v}                                                                           \
                        libnvinfer-plugin-dev=${v}                                                                      \
                        python3-libnvinfer=${v}                                                                         \
                        libnvinfer-dispatch8=${v}                                                                       \
                        libnvinfer-dispatch-dev=${v}                                                                    \
                        libnvinfer-lean8=${v}                                                                           \
                        libnvinfer-lean-dev=${v}                                                                        \
                        libnvinfer-vc-plugin8=${v}                                                                      \
                        libnvinfer-vc-plugin-dev=${v}                                                                   \
                        libnvinfer-headers-dev=${v}                                                                     \
                        libnvinfer-headers-plugin-dev=${v}

# Install PyPI packages
RUN python3 -m pip install --upgrade pip &&                     \
    python3 -m pip install --no-cache-dir setuptools==61.2.0    \
        Flask==2.1.0                                            \
        PyYAML==6.0                                             \
        requests==2.27.1                                        \
        gunicorn==20.1.0                                        \
        loguru==0.6.0                                           \
        prometheus-client==0.13.1

# Set environment and working directory
ENV TRT_LIBPATH /usr/lib/x86_64-linux-gnu
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${TRT_LIBPATH}"

# install python package
RUN python3 -m pip install --no-cache-dir protobuf==3.20.0
RUN python3 -m pip install --no-cache-dir pycuda==2022.1
RUN python3 -m pip install torch==1.13.1 torchvision==0.14.1 --extra-index-url https://download.pytorch.org/whl/cu11
RUN python3 -m pip install --no-cache-dir nvidia-pyindex==1.0.9
RUN python3 -m pip install --no-cache-dir onnx-graphsurgeon==0.3.19
RUN python3 -m pip install --no-cache-dir onnxruntime-gpu==1.12.1
RUN python3 -m pip install --no-cache-dir onnx-simplifier==0.3.10
RUN python3 -m pip install --no-cache-dir pydantic==1.10.2
RUN python3 -m pip install --no-cache-dir nvidia-ml-py

RUN python3 -m pip install --no-cache-dir timm==0.6.7
RUN python3 -m pip install efficientnet_pytorch==0.7.1
RUN python3 -m pip install --no-cache-dir importlib_metadata
RUN python3 -m pip install --no-cache-dir transformers==4.19.2

RUN python3 -m pip install --no-cache-dir psutil==5.9.2 pynvml==11.4.1
RUN python3 -m pip install --no-cache-dir tabulate pandas matplotlib==3.6.1 seaborn
RUN python3 -m pip install  --no-cache-dir tqdm boto3 requests regex sentencepiece sacremoses

WORKDIR /workspace

CMD ["bash"]
