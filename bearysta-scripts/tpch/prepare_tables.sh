if [ ! -d "gen_$1" ]; then 
    mkdir gen_$1
    pushd tpch-dbgen
    make
    ./dbgen -s $1 -f
    sed -i 's/.$//' *.tbl

    mv *.tbl ../gen_$1/

    for q in 1 5 11; do
        DSS_QUERY=../query_templates ./qgen $q -s $1 > ../gen_$1/$q.sql
        dos2unix -o ../gen_$1/$q.sql
    done;

    popd

    echo "USER admin omnisci {" > gen_$1/queries.sql

    for q in 1 5 11; do
        cat gen_$1/$q.sql >> gen_$1/queries.sql
        cat gen_$1/$q.sql >> gen_$1/queries.sql
        cat gen_$1/$q.sql >> gen_$1/queries.sql
    done;

    echo "}" >> gen_$1/queries.sql
fi

for f in gen_$1/*.tbl; do
    echo "$f lines count: $(wc -l < $f)"
done;

if [ ! -d "gen_$1/data-$2" ]; then
    mkdir -p gen_$1/data-$2 && ../../../omniscidb/build/bin/initdb --data gen_$1/data-$2

    python3 prepare_tables.py gen_$1 $2

    ../../../omniscidb/build/bin/omnisci_server --db-query-list=gen_$1/create_tables.sql --exit-after-warmup --data gen_$1/data-$2 --log-severity-clog=INFO
fi
