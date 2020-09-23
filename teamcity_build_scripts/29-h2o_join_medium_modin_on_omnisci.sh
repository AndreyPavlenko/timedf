#!/bin/bash

mkdir -p ${PWD}/tmp
python3 run_ibis_tests.py --env_name ${ENV_NAME} --env_check True --save_env True --python_version 3.7 -task benchmark         \
                          --ci_requirements "${PWD}/ci_requirements.yml"                                                       \
                          -bench_name h2o -data_file '${DATASETS_PWD}/h2o/J1_1e8_NA_0_0.csv'                                   \
                          -pandas_mode Modin_on_omnisci -ray_tmpdir ${PWD}/tmp                                                     \
                          -commit_omnisci ${BUILD_REVISION}                                                                    \
                          -commit_omniscripts ${BUILD_OMNISCRIPTS_REVISION}                                                    \
                          ${ADDITIONAL_OPTS}                                                                                   \
                          ${ADDITIONAL_OPTS_NIGHTLY}                                                                           \
                          ${DB_COMMON_OPTS} ${DB_H2O_OPTS}
