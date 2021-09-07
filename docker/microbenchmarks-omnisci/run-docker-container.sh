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

if [ -z "SHM_MEM" ]; then
    # use default shared memory size
    SHM_MEM=8gb
fi

docker run --shm-size $SHM_MEM --rm --name $CONTAINER_NAME $IMAGE_NAME \
    bash --login omniscripts/docker/microbenchmarks-omnisci/asv-runner.sh
