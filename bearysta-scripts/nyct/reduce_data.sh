#!/usr/bin/env bash

all_files=`ls /nfs/site/proj/scripting_tools/itocar/trips*.csv.gz`

for file in $all_files
do
    local_file=$(basename -- "$file")
    local_file=${local_file/trips/trips_reduced}
    echo Reducing $file "-->" $local_file
    gzip -cd $file | cut -d "," -f 3,11,12,20,25 | gzip -c - >$local_file
done
