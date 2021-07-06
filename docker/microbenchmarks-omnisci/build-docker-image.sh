#!/bin/bash -e

cd "`dirname \"$0\"`"

docker build -f microbenchmarks-omnisci.dockerfile -t microbenchmarks-omnisci \
    --build-arg https_proxy --build-arg http_proxy .

printf "\n\nTo run the benchmark execute:\n"
printf "\tdocker run --rm -v /path/to/output_bench_results:/bench_results microbenchmarks-omnisci\n
