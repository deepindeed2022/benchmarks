FROM nvcr.io/nvidia/pytorch:23.04-py3

RUN pip install ninja
RUN pip install -v -U git+https://github.com/facebookresearch/xformers.git@v0.0.21#egg=xformers
RUN pip install git+https://github.com/Dao-AILab/flash-attention.git@v2.1.0

WORKDIR /workspace
COPY . /workspace
