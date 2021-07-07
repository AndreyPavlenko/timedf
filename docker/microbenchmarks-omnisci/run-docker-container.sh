#!/bin/bash -e

# TODO: increase /dev/shm
docker run --rm --name microbenchmarks-omnisci-container microbenchmarks-omnisci \
    omniscripts/docker/microbenchmarks-omnisci/asv-runner.sh

# docker run --rm --name microbenchmarks-omnisci-container microbenchmarks-omnisci \
#    bash --login -c "export MODIN_BACKEND=omnisci \
# && export MODIN_EXPERIMENTAL=true \
# && export MODIN_TEST_DATASET_SIZE=big \
# && cd modin/asv_bench \
# && conda activate modin_on_omnisci \
# && asv run --launch-method=forkserver --config asv.conf.omnisci.json \
#    -b omnisci.benchmarks.TimeAppend --machine docker-omnisci -a repeat=3 \
#    --show-stderr --python=same"
