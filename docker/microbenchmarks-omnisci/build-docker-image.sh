#!/bin/bash -e

NAME=$1

if [ -z "$NAME" ]; then
    # use default name for docker image
    NAME=microbenchmarks-omnisci
fi

FIRST_STAGE_NAME=$NAME-intermediate

echo "first docker stage image name - $FIRST_STAGE_NAME"
echo "result image name - $NAME"

docker build -t $FIRST_STAGE_NAME -f omniscripts/docker/Dockerfile.omnisci-from-conda \
    --build-arg https_proxy --build-arg http_proxy omniscripts/docker

mkdir empty-context
docker build -t $NAME -f omniscripts/docker/microbenchmarks-omnisci/second-stage.dockerfile \
    --build-arg https_proxy --build-arg http_proxy --build-arg image_name=$FIRST_STAGE_NAME \
    --build-arg DB_COMMON_OPTS empty-context
