#!/usr/bin/env bash

all_files=`ls /localdisk/taxi_data_reduced/trips_reduced_*.csv`
cur_dir=`pwd`

for file in $all_files
do
    echo "Importing $file..."
    echo "COPY trips_reduced FROM '$file' WITH (header='false');" | /localdisk/amalakho/projects/omniscidb/build/bin/omnisql -p HyperInteractive --port 61274
done
