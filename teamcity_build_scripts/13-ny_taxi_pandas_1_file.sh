ROOT_DIR="${PWD}"
cd omniscripts
python3 taxi/taxibench_pandas.py -df 1 -i 5 -dp '/localdisk/benchmark_datasets/taxi/trips_*.csv.gz' -db-server=ansatlin07.an.intel.com -db-user=gashiman -db-pass=omniscidb -db-name=omniscidb -db-table=taxibench20m -commit ${BUILD_REVISION}
