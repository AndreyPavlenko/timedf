#!/bin/bash -e

export MODIN_BACKEND=omnisci
export MODIN_EXPERIMENTAL=true
export MODIN_TEST_DATASET_SIZE=big

cd modin/asv_bench
conda activate modin_on_omnisci

asv run  --launch-method=forkserver --config asv.conf.omnisci.json \
    -b omnisci.benchmarks.TimeAppend --machine docker-omnisci -a repeat=3 \
    --show-stderr --python=same

cd ../..
python omniscripts/report_asv_result.py \
    --result-path modin/asv_bench/.asv/results/<machine-name>/<json-name> \
    ${DB_COMMON_OPTS}
