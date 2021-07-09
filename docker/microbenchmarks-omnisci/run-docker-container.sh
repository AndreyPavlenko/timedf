#!/bin/bash -e

IMAGE_NAME=$1
CONTAINER_NAME=$2
SHM_MEM=$3

if [ -z "$IMAGE_NAME" ]; then
    # use default name for docker image
    IMAGE_NAME=microbenchmarks-omnisci
fi

if [ -z "$CONTAINER_NAME" ]; then
    # use default name for docker container
    CONTAINER_NAME=$IMAGE_NAME-container
fi

# TODO: increase /dev/shm
docker run --shm-size $SHM_MEM --rm --name $CONTAINER_NAME $IMAGE_NAME \
    bash --login omniscripts/docker/microbenchmarks-omnisci/asv-runner.sh

# docker run --rm --name $CONTAINER_NAME $IMAGE_NAME \
#    bash --login -c "export MODIN_BACKEND=omnisci \
# && export MODIN_EXPERIMENTAL=true \
# && export MODIN_TEST_DATASET_SIZE=big \
# && cd modin/asv_bench \
# && conda activate modin_on_omnisci \
# && asv run --launch-method=forkserver --config asv.conf.omnisci.json \
#    -b omnisci.benchmarks.TimeAppend --machine docker-omnisci -a repeat=3 \
#    --show-stderr --python=same"
