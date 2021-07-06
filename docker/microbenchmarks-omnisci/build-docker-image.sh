#!/bin/bash -e

docker build -t amyachev-test -f omniscripts/docker/Dockerfile.omnisci-from-conda --build-arg https_proxy --build-arg http_proxy omniscripts/docker
mkdir empty-context
docker build -t amyachev-test-stage2 -f omniscripts/docker/second-stage.dockerfile --build-arg https_proxy --build-arg http_proxy empty-context

docker run --rm --name amyachev-test-container amyachev-test-stage2 omniscripts/docker/microbenchmarks-omnisci/asv-runner.sh
#docker run --rm --name amyachev-test-container amyachev-test-stage2 bash --login -c "export MODIN_BACKEND=omnisci && export MODIN_EXPERIMENTAL=true && export MODIN_TEST_DATASET_SIZE=big && cd modin/asv_bench && conda activate modin_on_omnisci && asv run  --launch-method=forkserver --config asv.conf.omnisci.json -b omnisci.benchmarks.TimeAppend --machine docker-omnisci -a repeat=3 --show-stderr --python=same"#
