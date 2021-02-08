#!/bin/bash
set -xe
if [ ! -d "gen_$1" ]; then
    mkdir gen_$1
    pushd tpch-dbgen
    make
    ./dbgen -s $1 -f
    sed -ri 's/\|\s*$//' *.tbl

    mv *.tbl ../gen_$1/

    for q in 1 5 11; do
        DSS_QUERY=../query_templates ./qgen $q -s $1 >../gen_$1/$q.sql
    done

    popd
    pushd gen_$1/

    for q in 1 5 11; do
        dos2unix -o $q.sql
        echo "USER admin omnisci {" >q$q.sql
        echo "--itt_pause" >>q$q.sql

        cat $q.sql >> q$q.sql
        echo "--itt_resume" >> q$q.sql
        cat $q.sql >> q$q.sql
        cat $q.sql >> q$q.sql
        cat $q.sql >> q$q.sql
        cat $q.sql >> q$q.sql
        echo "--itt_pause" >> q$q.sql
    done;



    echo "USER admin omnisci {" > queries.sql
    echo "--itt_pause" >> queries.sql

    for q in 1 5 11; do
        cat $q.sql >> queries.sql
        echo "--itt_resume">> queries.sql
        cat $q.sql >> queries.sql
        cat $q.sql >> queries.sql
        echo "--itt_pause" >> queries.sql
    done;

    echo "}" >> queries.sql
    popd
fi

for f in gen_$1/*.tbl; do
    echo "$f lines count: $(wc -l < $f)"
done;

fragments=$(($2*$3))
if [ ! -d "gen_$1/data-$fragments" ]; then
    mkdir -p gen_$1/data-$fragments && ../../../omniscidb/build/bin/initdb --data gen_$1/data-$fragments

    python3 prepare_tables.py gen_$1 $2 $3

    ../../../omniscidb/build/bin/omnisci_server --db-query-list=gen_$1/create_tables.sql --exit-after-warmup --data gen_$1/data-$fragments
fi
