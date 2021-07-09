#!/bin/bash -e

# get performance of modin on omnisci
export MODIN_BACKEND=omnisci
export MODIN_EXPERIMENTAL=true
export MODIN_TEST_DATASET_SIZE=big

cd modin/asv_bench
conda activate modin_on_omnisci

asv run --launch-method=forkserver --config asv.conf.omnisci.json \
    -b omnisci.benchmarks.TimeAppend --machine docker-omnisci -a repeat=3 \
    --show-stderr --python=same --set-commit-hash HEAD
omnisci_result_name=`ls .asv/results/docker-omnisci/ | grep existing`

# get performance of pure pandas
export MODIN_ASV_USE_IMPL=pandas

asv run --launch-method=forkserver --config asv.conf.omnisci.json \
    -b omnisci.benchmarks.TimeAppend --machine docker-pandas -a repeat=3 \
    --show-stderr --python=same --set-commit-hash HEAD
pandas_result_name=`ls .asv/results/docker-pandas/ | grep existing`

# report modin on omnisci results
cd ../../omniscripts
python report_asv_result.py \
    --result-path modin/asv_bench/.asv/results/docker-omnisci/omnisci_result_name \
    $DB_COMMON_OPTS

# report pandas results
python report_asv_result.py \
    --result-path modin/asv_bench/.asv/results/docker-pandas/pandas_result_name \
    $DB_COMMON_OPTS
