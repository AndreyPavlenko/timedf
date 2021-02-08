#!/bin/bash
set -xe
bash prepare_tables.sh $scale $OMP_NUM_THREADS $slack
numactl $numa -C $cpus time $omnisci_server --config omnisci-bench-tpc-h.conf --data gen_$scale/data-$(($OMP_NUM_THREADS*$slack)) --db-query-list=gen_$scale/$query.sql $* 2>&1
