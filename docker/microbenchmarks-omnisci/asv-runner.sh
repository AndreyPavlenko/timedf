#!/bin/bash -e
set -o pipefail

# to resolve problem with proxy in docker for ray:
# https://github.com/modin-project/modin/issues/2608
PIP_CONF_PATH=pip.conf
cat << EOF > $PIP_CONF_PATH
[global]
proxy = `echo $http_proxy`
EOF
mkdir $HOME/.pip
mv $PIP_CONF_PATH $HOME/.pip/

cd modin/asv_bench

HASHFILE_PATH=./hashfile.txt
echo `git rev-parse HEAD` > $HASHFILE_PATH

unset http_proxy
unset https_proxy
export MODIN_TEST_DATASET_SIZE=Big
export MODIN_CPUS=44
export MODIN_MEMORY=128000000000

# benchmarking OmniSci backend
MODIN_BACKEND=omnisci MODIN_EXPERIMENTAL=true asv run HASHFILE:hashfile.txt \
    --show-stderr --launch-method=forkserver \
    --config asv.conf.omnisci.json -b ^omnisci -a repeat=3 -v

cp -r .asv/results /bench_results
