# coding: utf-8
import argparse
import gzip
import json
import os
import sys
import time
import traceback
import warnings
from timeit import default_timer as timer

import cloudpickle

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from server import OmnisciServer
from utils import (
    compare_dataframes,
    import_pandas_into_module_namespace,
    load_data_pandas,
    print_times,
)

warnings.filterwarnings("ignore")


# Dataset link
# https://rapidsai-data.s3.us-east-2.amazonaws.com/datasets/ipums_education2income_1970-2010.csv.gz


def etl_pandas(filename, columns_names, columns_types):
    etl_times = {
        "t_readcsv": 0.0,
        "t_where": 0.0,
        "t_arithm": 0.0,
        "t_fillna": 0.0,
        "t_drop": 0.0,
        "t_typeconvert": 0.0,
        "t_etl": 0.0,
    }

    t0 = timer()
    df = load_data_pandas(
        filename=filename,
        columns_names=columns_names,
        columns_types=columns_types,
        header=0,
        nrows=None,
        use_gzip=filename.endswith(".gz"),
        pd=main.__globals__["pd"],
    )
    etl_times["t_readcsv"] = timer() - t0

    t_etl_start = timer()

    keep_cols = [
        "YEAR0",
        "DATANUM",
        "SERIAL",
        "CBSERIAL",
        "HHWT",
        "CPI99",
        "GQ",
        "PERNUM",
        "SEX",
        "AGE",
        "INCTOT",
        "EDUC",
        "EDUCD",
        "EDUC_HEAD",
        "EDUC_POP",
        "EDUC_MOM",
        "EDUCD_MOM2",
        "EDUCD_POP2",
        "INCTOT_MOM",
        "INCTOT_POP",
        "INCTOT_MOM2",
        "INCTOT_POP2",
        "INCTOT_HEAD",
        "SEX_HEAD",
    ]
    t0 = timer()
    df = df[keep_cols]
    etl_times["t_drop"] += timer() - t0

    t0 = timer()
    df = df.query("INCTOT != 9999999")
    df = df.query("EDUC != -1")
    df = df.query("EDUCD != -1")
    etl_times["t_where"] += timer() - t0

    t0 = timer()
    df["INCTOT"] = df["INCTOT"] * df["CPI99"]
    etl_times["t_arithm"] += timer() - t0

    for column in keep_cols:
        t0 = timer()
        df[column] = df[column].fillna(-1)
        etl_times["t_fillna"] += timer() - t0

        t0 = timer()
        df[column] = df[column].astype("float64")
        etl_times["t_typeconvert"] += timer() - t0

    t0 = timer()
    y = df["EDUC"]
    X = df.drop(columns=["EDUC", "CPI99"])
    etl_times["t_drop"] += timer() - t0

    etl_times["t_etl"] = timer() - t_etl_start
    print("DataFrame shape:", X.shape)

    return df, X, y, etl_times


def etl_ibis(
    filename,
    columns_names,
    columns_types,
    database_name,
    table_name,
    omnisci_server_worker,
    delete_old_database,
    create_new_table,
    validation,
):
    etl_times = {
        "t_readcsv": 0.0,
        "t_where": 0.0,
        "t_arithm": 0.0,
        "t_fillna": 0.0,
        "t_pandas_drop": 0.0,
        "t_drop": 0.0,
        "t_typeconvert": 0.0,
        "t_etl": 0.0,
    }

    import ibis

    time.sleep(2)
    omnisci_server_worker.connect_to_server()

    omnisci_server_worker.create_database(
        database_name, delete_if_exists=delete_old_database
    )

    t0 = timer()

    omnisci_server_worker.connect_to_server(database=database_name)
    # Create table and import data
    if create_new_table:
        # Datafiles import
        t_import_pandas, t_import_ibis = omnisci_server_worker.import_data_by_ibis(
            table_name=table_name,
            data_files_names=filename,
            files_limit=1,
            columns_names=columns_names,
            columns_types=columns_types,
            header=0,
            nrows=None,
            compression_type=None,
            validation=validation,
        )

    etl_times["t_readcsv"] = t_import_pandas + t_import_ibis

    # Second connection - this is ibis's ipc connection for DML
    omnisci_server_worker.ipc_connect_to_server()
    db = omnisci_server_worker.database(database_name)
    table = db.table(table_name)

    t_etl_start = timer()

    keep_cols = [
        "YEAR0",
        "DATANUM",
        "SERIAL",
        "CBSERIAL",
        "HHWT",
        "CPI99",
        "GQ",
        "PERNUM",
        "SEX",
        "AGE",
        "INCTOT",
        "EDUC",
        "EDUCD",
        "EDUC_HEAD",
        "EDUC_POP",
        "EDUC_MOM",
        "EDUCD_MOM2",
        "EDUCD_POP2",
        "INCTOT_MOM",
        "INCTOT_POP",
        "INCTOT_MOM2",
        "INCTOT_POP2",
        "INCTOT_HEAD",
        "SEX_HEAD",
    ]

    if validation:
        keep_cols.append("id")

    table = table[keep_cols]
    etl_times["t_drop"] += timer() - t_etl_start

    # first, we do all filters and eliminate redundant fillna operations for EDUC and EDUCD
    t0 = timer()
    table = table[table.INCTOT != 9999999]
    table = table[table["EDUC"].notnull()]
    table = table[table["EDUCD"].notnull()]
    etl_times["t_where"] += timer() - t0

    t0 = timer()
    table = table.set_column("INCTOT", table["INCTOT"] * table["CPI99"])
    etl_times["t_arithm"] += timer() - t0

    cols = []
    # final fillna and casting for necessary columns
    for column in keep_cols:
        t0 = timer()
        cols.append(
            ibis.case()
            .when(table[column].notnull(), table[column])
            .else_(-1)
            .end()
            .cast("float64")
            .name(column)
        )
        etl_times["t_fillna"] += timer() - t0

    table = table.mutate(cols)

    df = table.execute()

    # here we use pandas to split table
    t0 = timer()
    y = df["EDUC"]
    X = df.drop(["EDUC", "CPI99"], axis=1)
    etl_times["t_pandas_drop"] = timer() - t0

    etl_times["t_etl"] = timer() - t_etl_start
    print("DataFrame shape:", X.shape)

    return df, X, y, etl_times


def mse(y_test, y_pred):
    return ((y_test - y_pred) ** 2).mean()


def cod(y_test, y_pred):
    y_bar = y_test.mean()
    total = ((y_test - y_bar) ** 2).sum()
    residuals = ((y_test - y_pred) ** 2).sum()
    return 1 - (residuals / total)


def ml(X, y, random_state, n_runs, train_size, optimizer):
    if optimizer == "intel":
        print("Intel optimized sklearn is used")
        from daal4py.sklearn.model_selection import train_test_split
        import daal4py.sklearn.linear_model as lm
    if optimizer == "stock":
        print("Stock sklearn is used")
        from sklearn.model_selection import train_test_split
        import sklearn.linear_model as lm
    else:
        print(
            f"Intel optimized and stock sklearn are supported. {optimizer} can't be recognized"
        )
        sys.exit(1)

    clf = lm.Ridge()

    mse_values, cod_values = [], []
    ml_times = {"t_split": 0.0, "t_ML": 0.0, "t_train": 0.0, "t_inference": 0.0}

    print("ML runs: ", n_runs)
    for i in range(n_runs):
        t0 = timer()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=train_size, random_state=random_state
        )
        ml_times["t_split"] += timer() - t0
        random_state += 777

        t0 = timer()
        model = clf.fit(X_train, y_train)
        ml_times["t_train"] += timer() - t0

        t0 = timer()
        y_pred = model.predict(X_test)
        ml_times["t_inference"] += timer() - t0

        mse_values.append(mse(y_test, y_pred))
        cod_values.append(cod(y_test, y_pred))

    ml_times["t_ML"] += ml_times["t_train"] + ml_times["t_inference"]

    mse_mean = sum(mse_values) / len(mse_values)
    cod_mean = sum(cod_values) / len(cod_values)
    mse_dev = pow(
        sum([(mse_value - mse_mean) ** 2 for mse_value in mse_values])
        / (len(mse_values) - 1),
        0.5,
    )
    cod_dev = pow(
        sum([(cod_value - cod_mean) ** 2 for cod_value in cod_values])
        / (len(cod_values) - 1),
        0.5,
    )

    return mse_mean, cod_mean, mse_dev, cod_dev, ml_times


def main():
    omniscript_path = os.path.dirname(__file__)
    args = None
    omnisci_server_worker = None

    parser = argparse.ArgumentParser(
        description="Run cencus benchmark on Ibis and Pandas."
    )
    optional = parser._action_groups.pop()
    required = parser.add_argument_group("required arguments")
    parser._action_groups.append(optional)

    required.add_argument(
        "-f", "--file", dest="file", help="A datafile that should be loaded.",
    )
    optional.add_argument(
        "-df",
        "--dfiles_num",
        dest="dfiles_num",
        default=1,
        type=int,
        help="Number of datafiles to input into database for processing.",
    )
    required.add_argument(
        "--omnisci_server_worker",
        dest="omnisci_server_worker",
        default="server_worker.pickled",
        help="File with pickled omnisci_server_worker representation.",
    )
    optional.add_argument(
        "--result_file",
        dest="result_file",
        default="taxi_results.json",
        help="File to which the results will be written.",
    )
    # Omnisci server parameters
    optional.add_argument(
        "-db",
        "--database_name",
        dest="database_name",
        default="omnisci",
        help="Database name to use in omniscidb server.",
    )
    optional.add_argument(
        "-t",
        "--table",
        dest="table",
        default="benchmark_table",
        help="Table name name to use in omniscidb server.",
    )
    # Ibis parameters
    optional.add_argument("-dnd", action="store_true", help="Do not delete old table.")
    optional.add_argument(
        "-dni",
        action="store_true",
        help="Do not create new table and import any data from CSV files.",
    )
    # Benchmark parameters
    optional.add_argument(
        "-val",
        dest="validation",
        action="store_true",
        help="validate queries results (by comparison with Pandas queries results).",
    )
    optional.add_argument(
        "-o",
        "--optimizer",
        choices=["intel", "stock"],
        dest="optimizer",
        default="intel",
        help="Which optimizer is used",
    )
    # Benchmark parameters
    optional.add_argument(
        "-no_ibis",
        action="store_true",
        help="Do not run Ibis benchmark, run only Pandas (or Modin) version",
    )
    optional.add_argument(
        "-pandas_mode",
        choices=["pandas", "modin_on_ray", "modin_on_dask", "modin_on_python"],
        default="pandas",
        help="Specifies which version of Pandas to use: plain Pandas, Modin runing on Ray or on Dask",
    )
    optional.add_argument(
        "-ray_tmpdir",
        default="/tmp",
        help="Location where to keep Ray plasma store. It should have enough space to keep -ray_memory",
    )
    optional.add_argument(
        "-ray_memory",
        default=200 * 1024 * 1024 * 1024,
        help="Size of memory to allocate for Ray plasma store",
    )
    optional.add_argument(
        "-no_ml",
        action="store_true",
        help="Do not run machine learning benchmark, only ETL part",
    )
    optional.add_argument(
        "-q3_full",
        action="store_true",
        help="Execute q3 query correctly (script execution time will be increased).",
    )

    args = parser.parse_args()

    ignored_args = {
        "q3_full": args.q3_full,
        "dfiles_num": args.dfiles_num,
    }
    if args.no_ibis:
        ignored_args["omnisci_server_worker"] = args.omnisci_server_worker
        ignored_args["dnd"] = args.dnd
        ignored_args["dni"] = args.dni
    warnings.warn(f"Parameters {ignored_args} are irnored", RuntimeWarning)

    args.file = args.file.replace("'", "")

    # ML specific
    N_RUNS = 50
    TRAIN_SIZE = 0.9
    RANDOM_STATE = 777

    columns_names = [
        "YEAR0",
        "DATANUM",
        "SERIAL",
        "CBSERIAL",
        "HHWT",
        "CPI99",
        "GQ",
        "QGQ",
        "PERNUM",
        "PERWT",
        "SEX",
        "AGE",
        "EDUC",
        "EDUCD",
        "INCTOT",
        "SEX_HEAD",
        "SEX_MOM",
        "SEX_POP",
        "SEX_SP",
        "SEX_MOM2",
        "SEX_POP2",
        "AGE_HEAD",
        "AGE_MOM",
        "AGE_POP",
        "AGE_SP",
        "AGE_MOM2",
        "AGE_POP2",
        "EDUC_HEAD",
        "EDUC_MOM",
        "EDUC_POP",
        "EDUC_SP",
        "EDUC_MOM2",
        "EDUC_POP2",
        "EDUCD_HEAD",
        "EDUCD_MOM",
        "EDUCD_POP",
        "EDUCD_SP",
        "EDUCD_MOM2",
        "EDUCD_POP2",
        "INCTOT_HEAD",
        "INCTOT_MOM",
        "INCTOT_POP",
        "INCTOT_SP",
        "INCTOT_MOM2",
        "INCTOT_POP2",
    ]
    columns_types = [
        "int64",
        "int64",
        "int64",
        "float64",
        "int64",
        "float64",
        "int64",
        "float64",
        "int64",
        "int64",
        "int64",
        "int64",
        "int64",
        "int64",
        "int64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
    ]

    db_reporter = None

    try:

        import_pandas_into_module_namespace(
            namespace=main.__globals__,
            mode=args.pandas_mode,
            ray_tmpdir=args.ray_tmpdir,
            ray_memory=args.ray_memory,
        )

        etl_times_ibis = None
        if not args.no_ibis:
            X_ibis, y_ibis, etl_times_ibis = etl_ibis(
                filename=args.file,
                columns_names=columns_names,
                columns_types=columns_types,
                database_name=args.database_name,
                table_name=args.table,
                omnisci_server_worker=cloudpickle.load(
                    open(args.omnisci_server_worker, "rb")
                ),
                delete_old_database=not args.dnd,
                create_new_table=not args.dni,
                validation=args.val,
            )

            omnisci_server_worker.terminate()
            omnisci_server_worker = None

            print_times(etl_times_ibis, "Ibis", db_reporter)

            if not args.no_ml:
                mse_mean, cod_mean, mse_dev, cod_dev, ml_times = ml(
                    X_ibis, y_ibis, RANDOM_STATE, N_RUNS, TRAIN_SIZE, args.optimizer
                )
                print_times(ml_times, "Ibis")
                print("mean MSE ± deviation: {:.9f} ± {:.9f}".format(mse_mean, mse_dev))
                print("mean COD ± deviation: {:.9f} ± {:.9f}".format(cod_mean, cod_dev))

        X, y, etl_times = etl_pandas(
            args.file, columns_names=columns_names, columns_types=columns_types
        )

        print_times(etl_times=etl_times, backend=args.pandas_mode)
        etl_times["Backend"] = args.pandas_mode

        if not args.no_ml:
            mse_mean, cod_mean, mse_dev, cod_dev, ml_times = ml(
                X, y, RANDOM_STATE, N_RUNS, TRAIN_SIZE, args.optimizer
            )
            print_times(ml_times, args.pandas_mode)
            print("mean MSE ± deviation: {:.9f} ± {:.9f}".format(mse_mean, mse_dev))
            print("mean COD ± deviation: {:.9f} ± {:.9f}".format(cod_mean, cod_dev))

        if args.validation:
            compare_dataframes(ibis_df=(X_ibis, y_ibis), pandas_df=(X, y),
                               pd=main.__globals__["pd"])

        with open(args.result_file, "w") as json_file:
            json.dump([etl_times_ibis, etl_times], json_file)
    except Exception:
        traceback.print_exc(file=sys.stdout)
        sys.exit(1)
    finally:
        if omnisci_server_worker:
            omnisci_server_worker.terminate()


if __name__ == "__main__":
    main()
