#!/usr/bin/env bash

all_files=`ls trips_reduced_*.csv`
cur_dir=`pwd`

cat trips_reduced-header.csv >trips_reduced-full.csv
for file in $all_files
do
    echo "adding $file..."
    cat $file >>trips_reduced-full.csv
done
