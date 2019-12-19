ROOT_DIR="${PWD}"
cd omniscripts/taxi
python3 taxibench_ibis.py "${ROOT_DIR}"/omniscidb/build/bin/omnisci_server --port 61274 -df 20 -dp '/localdisk/benchmark_datasets/taxi/trips_xa{a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t}.csv.gz' -i 5 -db-server=ansatlin07.an.intel.com -db-user=gashiman -db-pass=omniscidb -db-name=omniscidb -db-table=taxibench -commit ${BUILD_REVISION}
