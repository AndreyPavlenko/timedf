import argparse
import json
import os
import sys
import time
import traceback
import warnings

import cloudpickle
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils import (
    compare_dataframes,
    files_names_from_pattern,
    import_pandas_into_module_namespace,
    load_data_pandas,
    print_times,
)


def validation_prereqs(
    omnisci_server_worker, data_files_names, files_limit, columns_names
):
    return omnisci_server_worker.import_data_by_pandas(
        data_files_names=data_files_names,
        files_limit=files_limit,
        columns_names=columns_names,
    )


def run_queries(queries, args, etl_times):
    for query_number, (query_name, query_func) in enumerate(queries.items()):
        print("Running query number:", query_number + 1)
        exec_time = int(round(query_func(**args) * 1000, 3))
        etl_times[query_name] = exec_time
        print("Query", query_number + 1, "Exec time (ms):", exec_time)
    return etl_times


# Queries definitions
def q1_ibis(
    table, df_pandas, queries_validation_results, queries_validation_flags, val
):
    t_query = 0
    t0 = time.time()
    q1_output_ibis = (
        table.groupby("cab_type")
        .count()
        .sort_by("cab_type")["cab_type", "count"]
        .execute()
    )
    t_query += time.time() - t0

    if val and not queries_validation_flags["q1"]:
        print("Validating query 1 results ...")

        queries_validation_flags["q1"] = True

        q1_output_pd = df_pandas.groupby("cab_type")["cab_type"].count()

        # Casting of Pandas q1 output to Pandas.DataFrame type, which is compartible with
        # Ibis q1 output
        q1_output_pd_df = q1_output_pd.to_frame()
        q1_output_pd_df.loc[:, "count"] = q1_output_pd_df.loc[:, "cab_type"]
        q1_output_pd_df["cab_type"] = q1_output_pd_df.index
        q1_output_pd_df.index = [i for i in range(len(q1_output_pd_df))]

        queries_validation_results["q1"] = compare_dataframes(
            ibis_df=q1_output_pd_df, pandas_df=q1_output_ibis, pd=main.__globals__["pd"]
        )
        if queries_validation_results["q1"]:
            print("q1 results are validated!")

    return t_query


def q2_ibis(
    table, df_pandas, queries_validation_results, queries_validation_flags, val
):
    t_query = 0
    t0 = time.time()
    q2_output_ibis = (
        table.groupby("passenger_count")
        .aggregate(total_amount=table.total_amount.mean())[
            ["passenger_count", "total_amount"]
        ]
        .execute()
    )
    t_query += time.time() - t0

    if val and not queries_validation_flags["q2"]:
        print("Validating query 2 results ...")

        queries_validation_flags["q2"] = True

        q2_output_pd = df_pandas.groupby("passenger_count", as_index=False).mean()[
            ["passenger_count", "total_amount"]
        ]

        queries_validation_results["q2"] = compare_dataframes(
            pandas_df=q2_output_pd, ibis_df=q2_output_ibis, pd=main.__globals__["pd"]
        )
        if queries_validation_results["q2"]:
            print("q2 results are validated!")

    return t_query


def q3_ibis(
    table, df_pandas, queries_validation_results, queries_validation_flags, val
):
    t_query = 0
    t0 = time.time()
    q3_output_ibis = (
        table.groupby(
            [
                table.passenger_count,
                table.pickup_datetime.year().name("pickup_datetime"),
            ]
        )
        .aggregate(count=table.passenger_count.count())
        .execute()
    )
    t_query += time.time() - t0

    if val and not queries_validation_flags["q3"]:
        print("Validating query 3 results ...")

        queries_validation_flags["q3"] = True

        transformed = df_pandas[["passenger_count", "pickup_datetime"]].transform(
            {
                "passenger_count": lambda x: x,
                "pickup_datetime": lambda x: pd.DatetimeIndex(x).year,
            }
        )
        q3_output_pd = transformed.groupby(["passenger_count", "pickup_datetime"])[
            ["passenger_count", "pickup_datetime"]
        ].count()["passenger_count"]

        # Casting of Pandas q3 output to Pandas.DataFrame type, which is compartible with
        # Ibis q3 output
        q3_output_pd_df = q3_output_pd.to_frame()
        count_df = q3_output_pd_df.loc[:, "passenger_count"].copy()
        q3_output_pd_df["passenger_count"] = q3_output_pd.index.droplevel(
            level="pickup_datetime"
        )
        q3_output_pd_df["pickup_datetime"] = q3_output_pd.index.droplevel(
            level="passenger_count"
        )
        q3_output_pd_df = q3_output_pd_df.astype({"pickup_datetime": "int32"})
        q3_output_pd_df.loc[:, "count"] = count_df
        q3_output_pd_df.index = [i for i in range(len(q3_output_pd_df))]

        queries_validation_results["q3"] = compare_dataframes(
            pandas_df=q3_output_pd_df, ibis_df=q3_output_ibis, pd=main.__globals__["pd"]
        )
        if queries_validation_results["q3"]:
            print("q3 results are validated!")

    return t_query


def q4_ibis(
    table, df_pandas, queries_validation_results, queries_validation_flags, val
):
    t_query = 0
    t0 = time.time()
    q4_ibis_sized = table.groupby(
        [
            table.passenger_count,
            table.pickup_datetime.year().name("pickup_datetime"),
            table.trip_distance.cast("int64").name("trip_distance"),
        ]
    ).size()
    q4_output_ibis = q4_ibis_sized.sort_by(
        [("pickup_datetime", True), ("count", False)]
    ).execute()
    t_query += time.time() - t0

    if val and not queries_validation_flags["q4"]:
        print("Validating query 4 results ...")

        queries_validation_flags["q4"] = True

        q4_pd_sized = (
            df_pandas[["passenger_count", "pickup_datetime", "trip_distance"]]
            .transform(
                {
                    "passenger_count": lambda x: x,
                    "pickup_datetime": lambda x: pd.DatetimeIndex(x).year,
                    "trip_distance": lambda x: x.astype("int64", copy=False),
                }
            )
            .groupby(["passenger_count", "pickup_datetime", "trip_distance"])
            .size()
            .reset_index()
        )

        q4_output_pd = q4_pd_sized.sort_values(
            by=["pickup_datetime", 0], ascending=[True, False]
        )

        # Casting of Pandas q4 output to Pandas.DataFrame type, which is compartible with
        # Ibis q4 output
        q4_output_pd = q4_output_pd.astype({"pickup_datetime": "int32"})
        q4_output_pd.columns = [
            "passenger_count",
            "pickup_datetime",
            "trip_distance",
            "count",
        ]
        q4_output_pd.index = [i for i in range(len(q4_output_pd))]

        # compare_result_1 and compare_result_2 are the results of comparison of q4 sorted columns
        compare_result_1 = compare_dataframes(
            pandas_df=q4_output_pd["pickup_datetime"],
            ibis_df=q4_output_ibis["pickup_datetime"],
            pd=main.__globals__["pd"],
        )
        compare_result_2 = compare_dataframes(
            pandas_df=q4_output_pd["count"],
            ibis_df=q4_output_ibis["count"],
            pd=main.__globals__["pd"],
        )

        # compare_result_3 is the result of q4 output table all elements presence check
        q4_output_ibis_validation = q4_ibis_sized.sort_by(
            [
                ("pickup_datetime", True),
                ("count", False),
                ("trip_distance", True),
                ("passenger_count", True),
            ]
        ).execute()
        q4_output_pd_valid = q4_pd_sized.sort_values(
            by=["trip_distance", "passenger_count"]
        ).sort_values(by=["pickup_datetime", 0], ascending=[True, False])
        q4_output_pd_valid = q4_output_pd_valid.astype({"pickup_datetime": "int32"})
        q4_output_pd_valid.columns = [
            "passenger_count",
            "pickup_datetime",
            "trip_distance",
            "count",
        ]
        q4_output_pd_valid.index = [i for i in range(len(q4_output_pd))]

        compare_result_3 = compare_dataframes(
            pandas_df=q4_output_pd_valid,
            ibis_df=q4_output_ibis_validation,
            pd=main.__globals__["pd"],
        )

        queries_validation_results["q4"] = (
            compare_result_1 and compare_result_2 and compare_result_3
        )
        if queries_validation_results["q4"]:
            print("q4 results are validated!")

    return t_query


def etl_ibis(
    filename,
    files_limit,
    columns_names,
    columns_types,
    database_name,
    table_name,
    omnisci_server_worker,
    delete_old_database,
    create_new_table,
    val,
):

    queries = {
        "Query1": q1_ibis,
        "Query2": q2_ibis,
        "Query3": q3_ibis,
        "Query4": q4_ibis,
    }
    etl_times = {x: 0.0 for x in queries.keys()}

    queries_validation_results = {"q%s" % i: False for i in range(1, 5)}
    queries_validation_flags = {"q%s" % i: False for i in range(1, 5)}

    conn = omnisci_server_worker.connect_to_server()

    data_files_names = files_names_from_pattern(filename)

    if len(data_files_names) == 0:
        print("Could not find any data files matching ", filename)
        sys.exit(2)

    omnisci_server_worker.create_database(
        database_name, delete_if_exists=delete_old_database
    )

    conn = omnisci_server_worker.connect_to_server(database=database_name)
    if create_new_table:
        # TODO t_import_pandas, t_import_ibis = omnisci_server_worker.import_data_by_ibis
        t0 = time.time()
        omnisci_server_worker.import_data(
            table_name=table_name,
            data_files_names=data_files_names,
            files_limit=files_limit,
            columns_names=columns_names,
            columns_types=columns_types,
            header=False,
        )
        etl_times["t_readcsv"] = time.time() - t0
        # etl_times["t_readcsv"] = t_import_pandas + t_import_ibis

    db = conn.database(database_name)
    table = db.table(table_name)

    df_pandas = None
    if val:
        df_pandas = validation_prereqs(
            omnisci_server_worker, data_files_names, files_limit, columns_names
        )

    queries_args = {
        "table": table,
        "df_pandas": df_pandas,
        "queries_validation_results": queries_validation_results,
        "queries_validation_flags": queries_validation_flags,
        "val": val,
    }
    return run_queries(queries=queries, args=queries_args, etl_times=etl_times)


# SELECT cab_type,
#       count(*)
# FROM trips
# GROUP BY cab_type;
# @hpat.jit fails with Invalid use of Function(<ufunc 'isnan'>) with argument(s) of type(s): (StringType), even when dtype is provided
def q1_pandas(df):
    t0 = time.time()
    df.groupby("cab_type").count()
    return time.time() - t0


# SELECT passenger_count,
#       count(total_amount)
# FROM trips
# GROUP BY passenger_count;
def q2_pandas(df):
    t0 = time.time()
    df.groupby("passenger_count", as_index=False).count()[
        ["passenger_count", "total_amount"]
    ]
    return time.time() - t0


# SELECT passenger_count,
#       EXTRACT(year from pickup_datetime) as year,
#       count(*)
# FROM trips
# GROUP BY passenger_count,
#         year;
def q3_pandas(df):
    t0 = time.time()
    transformed = df.applymap(lambda x: x.year if hasattr(x, "year") else x)
    transformed.groupby(["pickup_datetime", "passenger_count"], as_index=False).count()[
        "passenger_count"
    ]
    return time.time() - t0


# SELECT passenger_count,
#       EXTRACT(year from pickup_datetime) as year,
#       round(trip_distance) distance,
#       count(*) trips
# FROM trips
# GROUP BY passenger_count,
#         year,
#         distance
# ORDER BY year,
#         trips desc;
def q4_pandas(df):
    t0 = time.time()
    transformed = df.applymap(
        lambda x: x.year
        if hasattr(x, "year")
        else round(x)
        if isinstance(x, (int, float)) and not np.isnan(x)
        else x
    )[["passenger_count", "pickup_datetime", "trip_distance"]].groupby(
        ["passenger_count", "pickup_datetime", "trip_distance"]
    )
    (
        transformed.count()
        .reset_index()
        .sort_values(by=["pickup_datetime", "trip_distance"], ascending=[True, False])
    )
    return time.time() - t0


def etl_pandas(
    filename, files_limit, columns_names, columns_types,
):
    queries = {
        "Query1": q1_pandas,
        "Query2": q2_pandas,
        "Query3": q3_pandas,
        "Query4": q4_pandas,
    }
    etl_times = {x: 0.0 for x in queries.keys()}

    t0 = time.time()
    df_from_each_file = [
        load_data_pandas(
            filename=f,
            columns_names=columns_names,
            header=0,
            nrows=1000,
            use_gzip=f.endswith(".gz"),
            parse_dates=["pickup_datetime", "dropoff_datetime",],
            pd=main.__globals__["pd"],
        )
        for f in filename
    ]
    concatenated_df = pd.concat(df_from_each_file, ignore_index=True)
    etl_times["t_readcsv"] = time.time() - t0

    queries_args = {"df": concatenated_df}
    return run_queries(queries=queries, args=queries_args, etl_times=etl_times)


def main():

    parser = argparse.ArgumentParser(
        description="Run taxi benchmark on Ibis and Pandas"
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
        choices=["Pandas", "Modin_on_ray", "Modin_on_dask"],
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
        "optimizer": args.optimizer,
        "q3_full": args.q3_full,
        "no_ml": args.no_ml,
    }
    if args.no_ibis:
        ignored_args["omnisci_server_worker"] = args.omnisci_server_worker
        ignored_args["dnd"] = args.dnd
        ignored_args["dni"] = args.dni
    warnings.warn(f"Parameters {ignored_args} are irnored", RuntimeWarning)

    args.file = args.file.replace("'", "")

    columns_names = [
        "trip_id",
        "vendor_id",
        "pickup_datetime",
        "dropoff_datetime",
        "store_and_fwd_flag",
        "rate_code_id",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
        "trip_distance",
        "fare_amount",
        "extra",
        "mta_tax",
        "tip_amount",
        "tolls_amount",
        "ehail_fee",
        "improvement_surcharge",
        "total_amount",
        "payment_type",
        "trip_type",
        "pickup",
        "dropoff",
        "cab_type",
        "precipitation",
        "snow_depth",
        "snowfall",
        "max_temperature",
        "min_temperature",
        "average_wind_speed",
        "pickup_nyct2010_gid",
        "pickup_ctlabel",
        "pickup_borocode",
        "pickup_boroname",
        "pickup_ct2010",
        "pickup_boroct2010",
        "pickup_cdeligibil",
        "pickup_ntacode",
        "pickup_ntaname",
        "pickup_puma",
        "dropoff_nyct2010_gid",
        "dropoff_ctlabel",
        "dropoff_borocode",
        "dropoff_boroname",
        "dropoff_ct2010",
        "dropoff_boroct2010",
        "dropoff_cdeligibil",
        "dropoff_ntacode",
        "dropoff_ntaname",
        "dropoff_puma",
    ]

    columns_types = [
        "int64",
        "int64",
        "timestamp",
        "timestamp",
        "string",
        "int64",
        "float64",
        "float64",
        "float64",
        "float64",
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
        "int64",
        "float64",
        "string",
        "string",
        "string",
        "float64",
        "int64",
        "float64",
        "int64",
        "int64",
        "float64",
        "float64",
        "float64",
        "float64",
        "string",
        "float64",
        "float64",
        "string",
        "string",
        "string",
        "float64",
        "float64",
        "float64",
        "float64",
        "string",
        "float64",
        "float64",
        "string",
        "string",
        "string",
        "float64",
    ]

    if not args.dfiles_num or args.dfiles_num <= 0:
        print("Bad number of data files specified: ", args.dfiles_num)
        sys.exit(1)
    try:
        import_pandas_into_module_namespace(
            namespace=main.__globals__,
            mode=args.pandas_mode,
            ray_tmpdir=args.ray_tmpdir,
            ray_memory=args.ray_memory,
        )

        etl_times_ibis = None
        if not args.no_ibis:
            etl_times_ibis = etl_ibis(
                filename=args.file,
                files_limit=args.dfiles_num,
                columns_names=columns_names,
                columns_types=columns_types,
                database_name=args.database_name,
                table_name=args.table,
                omnisci_server_worker=cloudpickle.load(
                    open(args.omnisci_server_worker, "rb")
                ),
                delete_old_database=not args.dnd,
                create_new_table=not args.dni,
                val=args.validation,
            )

            print_times(etl_times=etl_times_ibis, backend="Ibis")
            etl_times_ibis["Backend"] = "Ibis"

        pandas_files_limit = 1
        filename = files_names_from_pattern(args.file)[:pandas_files_limit]
        etl_times = etl_pandas(
            filename=filename,
            files_limit=pandas_files_limit,
            columns_names=columns_names,
            columns_types=columns_types,
        )

        print_times(etl_times=etl_times, backend=args.pandas_mode)
        etl_times["Backend"] = args.pandas_mode

        with open(args.result_file, "w") as json_file:
            json.dump([etl_times_ibis, etl_times], json_file)
    except Exception:
        traceback.print_exc(file=sys.stdout)
        sys.exit(1)


if __name__ == "__main__":
    main()
