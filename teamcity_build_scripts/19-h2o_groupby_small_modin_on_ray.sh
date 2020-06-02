#!/bin/bash

mkdir -p ${PWD}/tmp
python3 run_ibis_tests.py --env_name ${ENV_NAME} --env_check True --save_env True --python_version 3.7 -task benchmark         \
                          --ci_requirements "${PWD}/ci_requirements.yml"                                                       \
                          --ibis_path "${PWD}/../ibis/"                                                                        \
                          -executable "${PWD}/../omniscidb/build/bin/omnisci_server"                                           \
                          -bench_name h2o -data_file '${DATASETS_PWD}/h2o/G1_1e7_1e2_0_0.csv'                                  \
                          -pandas_mode Modin_on_ray -ray_tmpdir ${PWD}/tmp                                                     \
                          -commit_omnisci ${BUILD_REVISION}                                                                    \
                          -commit_ibis ${BUILD_IBIS_REVISION}                                                                  \
                          -commit_omniscripts ${BUILD_OMNISCRIPTS_REVISION}                                                    \
                          ${ADDITIONAL_OPTS}                                                                                   \
                          ${ADDITIONAL_OPTS_NIGHTLY}                                                                           \
                          ${DB_COMMON_OPTS} ${DB_H2O_OPTS}
