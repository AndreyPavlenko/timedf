ROOT_DIR="${PWD}"
cd omniscripts/santander
python3 santander_ibis.py -e="${ROOT_DIR}"/omniscidb/build/bin/omnisql --port 61274 -dp '/localdisk/benchmark_datasets/santander/train.csv.gz' -i 5 -db-server=ansatlin07.an.intel.com -db-user=gashiman -db-pass=omniscidb -db-name=omniscidb -db-table=santander_ibis -commit ${BUILD_REVISION}