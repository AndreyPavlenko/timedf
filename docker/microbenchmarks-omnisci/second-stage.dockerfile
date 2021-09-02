# Create images from this container like this (up from modin and omniscripts repo roots):
#
# tar cf omniscripts/docker/modin-and-omniscripts.tar modin omniscripts
#
# omniscripts/docker/microbenchmarks-omnisci/build-docker-image.sh

ARG image_name=microbenchmarks-omnisci-intermediate
FROM ${image_name}:latest

# include also workaround for incompatible ray
RUN conda activate modin_on_omnisci \
 && pip install git+https://github.com/airspeed-velocity/asv.git@ef016e233cb9a0b19d517135104f49e0a3c380e9 \
 && conda uninstall ray-core -c conda-forge \
 && conda install mysql mysql-connector-python -c conda-forge \
 && pip install "ray[default]>=1.4"

ARG HOST_NAME=docker
ENV HOST_NAME ${HOST_NAME}

# BUG: https://github.com/airspeed-velocity/asv/issues/944 in `asv machine`
# command in noninteractive mode;
RUN cd modin/asv_bench \
 && conda activate modin_on_omnisci \
 && asv machine --yes \
 && asv machine --machine ${HOST_NAME}-omnisci --yes \
 && asv machine --machine ${HOST_NAME}-pandas --yes

# There is no way to specify the data size for each benchmark yet -
# use the following workround.
RUN cd modin \
 && git apply ../omniscripts/docker/microbenchmarks-omnisci/timeout.patch

ARG DB_COMMON_OPTS
ENV DB_COMMON_OPTS ${DB_COMMON_OPTS}
