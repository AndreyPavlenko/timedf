#!/bin/bash -e

# it is supposed to run from $HOME/cron/

echo "\n\n\n"
echo `date`

cd "$(dirname "$0")"

CURRENT_FOLDER=cron-run-env
if [ -d $CURRENT_FOLDER ]; then
    # remove artefacts from previous cron run
    echo $CURRENT_FOLDER exist - removing
    rm -Rf $CURRENT_FOLDER
fi
mkdir $CURRENT_FOLDER
cd $CURRENT_FOLDER

git clone https://github.com/modin-project/modin.git
# need second-stage.dockerfile and helper scripts
git clone --branch docker-bench https://github.com/intel-ai/omniscripts.git
tar cf omniscripts/docker/modin-and-omniscripts.tar modin omniscripts

IMAGE_NAME=microbenchmarks-omnisci
INTERMEDIATE_IMAGE_NAME=$IMAGE_NAME-intermediate
CONTAINER_NAME=$IMAGE_NAME-container

# remove docker artefacts from previous cron run
docker ps -a | grep -q $CONTAINER_NAME && docker rm $CONTAINER_NAME \
    || echo docker container: \"$CONTAINER_NAME\" not found
docker images | grep $IMAGE_NAME && docker rmi $IMAGE_NAME \
    || echo docker image: \"$IMAGE_NAME\" not found
docker images | grep $INTERMEDIATE_IMAGE_NAME && docker rmi $INTERMEDIATE_IMAGE_NAME \
    || echo docker image: \"$INTERMEDIATE_IMAGE_NAME\" not found

./omniscripts/docker/microbenchmarks-omnisci/build-docker-image.sh $IMAGE_NAME
./omniscripts/docker/microbenchmarks-omnisci/run-docker-container.sh $IMAGE_NAME $CONTAINER_NAME
