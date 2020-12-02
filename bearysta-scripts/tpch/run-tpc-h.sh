#!/bin/bash
set -xe
bash prepare_tables.sh $scale $OMP_NUM_THREADS
numactl $numa -C $cpus $omnisci_server --config omnisci-bench-tpc-h.conf --data gen_$scale/data-$OMP_NUM_THREADS --db-query-list=gen_$scale/queries.sql 2>&1
