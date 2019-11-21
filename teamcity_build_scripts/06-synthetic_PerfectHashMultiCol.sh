ROOT_DIR="${PWD}"
cd omniscripts
python3 run_omnisci_benchmark.py -m synthetic -path="${ROOT_DIR}"/omniscidb/Benchmarks -u admin -p HyperInteractive -e "${ROOT_DIR}"/omniscidb/build/bin/omnisci_server --port 61274 -n omnisci -t perfect_hash_multi_col_benchmark -l perfect_hash_multi_col_test -nf 10 -sq PerfectHashMultiCol -i 5 -fs 5000000 -db-server=ansatlin07.an.intel.com -db-user=gashiman -db-pass=omniscidb -db-name=omniscidb -db-table=perfecthashmulticolbench -commit ${BUILD_REVISION}
