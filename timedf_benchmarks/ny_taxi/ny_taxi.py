# Original SQL queries can be found here https://tech.marksblogg.com/billion-nyc-taxi-rides-nvidia-pascal-titan-x-mapd.html
import argparse
from collections import OrderedDict
from itertools import islice
from pathlib import Path

from timedf import BenchmarkResults, BaseBenchmark, tm
from timedf.backend import pd, Backend
from timedf.benchmark_utils import load_data_pandas, load_data_modin_on_hdk, print_results


accepted_data_files_for_pandas_import_mode = ["trips_xaa", "trips_xab", "trips_xac"]

ny_taxi_data_files_sizes_MB = OrderedDict(
    {
        "trips_xaa.csv": 8000,
        "trips_xab.csv": 8100,
        "trips_xac.csv": 4200,
        "trips_xad.csv": 7300,
        "trips_xae.csv": 8600,
        "trips_xaf.csv": 8600,
        "trips_xag.csv": 8600,
        "trips_xah.csv": 8600,
        "trips_xai.csv": 8600,
        "trips_xaj.csv": 8600,
        "trips_xak.csv": 8700,
        "trips_xal.csv": 8700,
        "trips_xam.csv": 8600,
        "trips_xan.csv": 8600,
        "trips_xao.csv": 8600,
        "trips_xap.csv": 8600,
        "trips_xaq.csv": 8600,
        "trips_xar.csv": 8600,
        "trips_xas.csv": 8600,
        "trips_xat.csv": 8600,
    }
)

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
    "category",
    "timestamp",
    "timestamp",
    "category",
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
    "category",
    "float64",
    "category",
    "category",
    "category",
    "float64",
    "int64",
    "float64",
    "int64",
    "int64",
    "float64",
    "float64",
    "float64",
    "float64",
    "category",
    "float64",
    "float64",
    "category",
    "category",
    "category",
    "float64",
    "float64",
    "float64",
    "float64",
    "category",
    "float64",
    "float64",
    "category",
    "category",
    "category",
    "float64",
]


def check_hdk():
    return Backend.get_name() == "Modin_on_hdk"


def get_ny_taxi_dataset_size(dfiles_num):
    return sum(list(ny_taxi_data_files_sizes_MB.values())[:dfiles_num])


def run_queries(queries, q2df, output_for_validation=None):
    for query_name, query_func in queries.items():
        with tm.timeit(query_name):
            query_result = query_func(q2df[query_name])
        if output_for_validation is not None:
            assert len(query_result) == 2
            output_for_validation[query_name] = query_result[1]


# Queries definitions
# SELECT cab_type,
#       count(*)
# FROM trips
# GROUP BY cab_type;
def q1(df):
    if not check_hdk():
        q1_output = df.groupby("cab_type")["cab_type"].count()
    else:
        q1_output = df.groupby("cab_type").size()
        Backend.trigger_execution(q1_output)

    return q1_output


# SELECT passenger_count,
#       avg(total_amount)
# FROM trips
# GROUP BY passenger_count;
def q2(df):
    if not check_hdk():
        q2_output = df.groupby("passenger_count", as_index=False).mean(numeric_only=True)[
            ["passenger_count", "total_amount"]
        ]
    else:
        q2_output = df.groupby("passenger_count").agg({"total_amount": "mean"})
        Backend.trigger_execution(q2_output)

    return q2_output


# SELECT passenger_count,
#       extract(year from pickup_datetime) as pickup_year,
#       count(*)
# FROM trips
# GROUP BY passenger_count,
#         pickup_year;
def q3(df):
    if not check_hdk():
        transformed = pd.DataFrame(
            {
                "pickup_datetime": df["pickup_datetime"].dt.year,
                "passenger_count": df["passenger_count"],
            }
        )
        q3_output = transformed.groupby(
            ["pickup_datetime", "passenger_count"], as_index=False
        ).size()
    else:
        df["pickup_datetime"] = df["pickup_datetime"].dt.year
        q3_output = df.groupby(["passenger_count", "pickup_datetime"]).size()
        Backend.trigger_execution(q3_output)  # to trigger real execution

    return q3_output


# SELECT passenger_count,
#       extract(year from pickup_datetime) as pickup_year,
#       cast(trip_distance as int) AS distance,
#       count(*) AS the_count
# FROM trips
# GROUP BY passenger_count,
#         pickup_year,
#         distance
# ORDER BY pickup_year,
#         the_count desc;


# SQL query with sorting for results validation
# SELECT passenger_count,
#       extract(year from pickup_datetime) as pickup_year,
#       cast(trip_distance as int) AS distance,
#       count(*) AS the_count
# FROM agent_test_modin
# GROUP BY passenger_count,
#         pickup_year,
#         distance
# ORDER BY passenger_count, pickup_year, distance, the_count;
def q4(df):
    if not check_hdk():
        transformed = pd.DataFrame(
            {
                "passenger_count": df["passenger_count"],
                "pickup_datetime": df["pickup_datetime"].dt.year,
                "trip_distance": df["trip_distance"].astype("int64"),
            }
        )
        q4_output = (
            transformed.groupby(
                ["passenger_count", "pickup_datetime", "trip_distance"], as_index=False
            )
            .size()
            .sort_values(by=["pickup_datetime", "size"], ascending=[True, False])
        )
    else:
        df["pickup_datetime"] = df["pickup_datetime"].dt.year
        df["trip_distance"] = df["trip_distance"].astype("int64")
        q4_output = (
            df.groupby(["passenger_count", "pickup_datetime", "trip_distance"], sort=False)
            .size()
            .reset_index()
            .sort_values(by=["pickup_datetime", 0], ignore_index=True, ascending=[True, False])
        )
        Backend.trigger_execution(q4_output)  # to trigger real execution

    return q4_output


def etl(filename, columns_names, columns_types, output_for_validation):
    if check_hdk() and any(f.endswith(".gz") for f in filename):
        raise NotImplementedError(
            "Modin_on_hdk mode doesn't support import of compressed files yet"
        )

    with tm.timeit("t_readcsv"):
        if check_hdk():
            df_from_each_file = [
                load_data_modin_on_hdk(
                    filename=f,
                    columns_names=columns_names,
                    columns_types=columns_types,
                    parse_dates=["timestamp"],
                    pd=pd,
                )
                for f in filename
            ]
        else:
            df_from_each_file = [
                load_data_pandas(
                    filename=f,
                    columns_names=columns_names,
                    header=None,
                    nrows=None,
                    use_gzip=f.endswith(".gz"),
                    parse_dates=["pickup_datetime", "dropoff_datetime"],
                    pd=pd,
                )
                for f in filename
            ]

        concatenated_df = pd.concat(df_from_each_file, ignore_index=True)
        Backend.trigger_loading(concatenated_df)

    queries = {"Query1": q1, "Query2": q2, "Query3": q3, "Query4": q4}
    q2df = {
        # FIXME seems like such copy op can affect benchmark
        query_name: concatenated_df.copy() if check_hdk() else concatenated_df
        for query_name in queries
    }

    return run_queries(queries=queries, q2df=q2df, output_for_validation=output_for_validation)


def run_benchmark(parameters):
    if parameters["dfiles_num"] <= 0:
        raise ValueError(f"Bad number of data files specified: {parameters['dfiles_num']}")

    pd_queries_outputs = {} if parameters["validation"] else None

    n_files = parameters["dfiles_num"]
    dataset_root = Path(parameters["data_file"])
    filename = [str(dataset_root / s) for s in islice(ny_taxi_data_files_sizes_MB, n_files)]
    etl(
        filename=filename,
        columns_names=columns_names,
        columns_types=columns_types,
        output_for_validation=pd_queries_outputs,
    )


class Benchmark(BaseBenchmark):
    __params__ = ("validation", "dfiles_num")

    def add_benchmark_args(self, parser: argparse.ArgumentParser):
        parser.add_argument(
            "-validation",
            default=False,
            action="store_true",
            help="validate queries results (by comparison with Pandas queries results).",
        )
        parser.add_argument(
            "-dfiles_num",
            dest="dfiles_num",
            default=1,
            type=int,
            help="Number of datafiles to load into database for processing.",
        )

    def run_benchmark(self, params) -> BenchmarkResults:
        with tm.timeit("total"):
            run_benchmark(params)

        task2time = tm.get_results()
        print_results(task2time)

        return BenchmarkResults(
            task2time, params={"dataset_size": get_ny_taxi_dataset_size(params["dfiles_num"])}
        )

    def load_data(self, target_dir, reload=False):
        from timedf.tools.s3_load import download_folder

        download_folder(
            "modin-datasets",
            "taxi",
            target_dir,
            reload=reload,
            pattern="taxi/trips_xa[abcdefghijklmnopqrst].csv$",
        )
