. /usr/local/mapd-deps/mapd-deps.sh

export MAPD_WEB_PORT=61273
export MAPD_HTTP_PORT=61278
export MAPD_CALCITE_PORT=61279
export MAPD_TCP_PORT=61274

cd omniscidb/build
OMNISCI_PATH=`pwd`
./bin/omnisci_server data --port ${MAPD_TCP_PORT} --http-port 62078 --calcite-port 62079 --config omnisci.conf &
OMNISCI_SERVER_PID=$!
sleep 10

cd -
cd omniscripts/taxi
python3 ./taxibench.py -commit ${BUILD_REVISION} -sco -df 20 -t 5 -fs 5000000 -port ${MAPD_TCP_PORT} -dp '/localdisk/benchmark_datasets/taxi/*.csv.gz' -e ${OMNISCI_PATH}/bin/omnisql -db-server=ansatlin07.an.intel.com -db-user=gashiman -db-pass=omniscidb -db-name=omniscidb
EC=$?
kill -INT ${OMNISCI_SERVER_PID}

test ${EC} -eq 0 || echo Benchmark exited with error code ${EC}; exit ${EC}
