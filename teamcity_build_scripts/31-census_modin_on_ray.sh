#!/bin/bash

mkdir -p ${PWD}/tmp
python3 run_ibis_tests.py --env_name ${ENV_NAME} --env_check True --save_env True --python_version 3.7 -task benchmark         \
                          --ci_requirements "${PWD}/ci_requirements.yml"                                                       \
                          -executable "${PWD}/../omniscidb/build/bin/omnisci_server"                                           \
                          -table census -bench_name census -dfiles_num 1 -iterations 1                                         \
                          -data_file '${DATASETS_PWD}/census/ipums_education2income_1970-2010.csv.gz'                          \
                          -pandas_mode Modin_on_ray -ray_tmpdir /tmp                                                           \
                          -commit_omnisci ${BUILD_REVISION}                                                                    \
                          -commit_omniscripts ${BUILD_OMNISCRIPTS_REVISION}                                                    \
                          ${ADDITIONAL_OPTS}                                                                                   \
                          ${ADDITIONAL_OPTS_NIGHTLY}                                                                           \
                          ${DB_COMMON_OPTS} ${DB_CENSUS_OPTS}
