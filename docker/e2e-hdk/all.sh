#!/bin/bash -eu
# Set location to store datasets, 
# WARNING: paths need to be absolute, otherwise docker will not mount
# WARNING: don't store datasets in the same folder as dockerfile, to avoid long context loading during docker build
export DATASETS_ROOT=/localdisk/ekrivov/datasets
export RESULTS_DIR=$(readlink -m results)

# Archive omniscripts for the upload 
tar -cf omniscripts.tar  --exclude=e2e-hdk ../../.

# Build the image, use optional `--build-arg http_proxy=${http_proxy} --build-arg https_proxy=${https_proxy}` to configure proxy.
docker build -t modin-project/benchmarks-reproduce:latest -f ./Dockerfile .

# Download data
./load_data.sh

# Run experiments
./run_docker.sh