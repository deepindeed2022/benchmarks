set -o
#
# build env
#
TAG=build_benchmark_models:v1.0
NAME=model_convert
docker build -f docker/Dockerfile -t ${TAG} .
docker run -itd --net=host --ipc=host --cap-add SYS_ADMIN --gpus all  --name ${NAME} \
    -v ~/Repo:/build ${TAG} /bin/bash
docker exec -it ${NAME} /bin/bash
