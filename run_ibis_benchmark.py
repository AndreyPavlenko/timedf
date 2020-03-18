# coding: utf-8
import argparse
import gzip
import json
import os
import sys
import traceback

import cloudpickle
import mysql.connector

from environment import CondaEnvironment
from report import DbReport
from server import OmnisciServer
from server_worker import OmnisciServerWorker
from utils import compare_dataframes, import_pandas_into_module_namespace


def main():
    omniscript_path = os.path.dirname(__file__)
    args = None
    omnisci_server = None

    benchmarks = {
        "ny_taxi": os.path.join(omniscript_path, "taxi", "taxibench_pandas_ibis.py"),
        "santander": os.path.join(omniscript_path, "santander", "santander_ibis.py"),
        "census": os.path.join(omniscript_path, "census", "census_pandas_ibis.py"),
        "plasticc": os.path.join(
            omniscript_path, "plasticc", "plasticc_pandas_ibis.py"
        ),
    }

    parser = argparse.ArgumentParser(description="Run internal tests from ibis project")
    optional = parser._action_groups.pop()
    required = parser.add_argument_group("required arguments")
    parser._action_groups.append(optional)

    required.add_argument(
        "-bn",
        "--bench_name",
        dest="bench_name",
        choices=list(benchmarks.keys()),
        help="Benchmark name.",
    )
    required.add_argument(
        "-en", "--env_name", dest="env_name", help="Conda env name.",
    )
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
    optional.add_argument(
        "-it",
        "--iterations",
        dest="iterations",
        default=5,
        type=int,
        help="Number of iterations to run every query. Best result is selected.",
    )
    optional.add_argument("-dnd", action="store_true", help="Do not delete old table.")
    optional.add_argument(
        "-dni",
        action="store_true",
        help="Do not create new table and import any data from CSV files.",
    )
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
    # MySQL database parameters
    optional.add_argument(
        "-db-server",
        dest="db_server",
        default="localhost",
        help="Host name of MySQL server.",
    )
    optional.add_argument(
        "-db-port",
        dest="db_port",
        default=3306,
        type=int,
        help="Port number of MySQL server.",
    )
    optional.add_argument(
        "-db-user",
        dest="db_user",
        help="Username to use to connect to MySQL database. "
             "If user name is specified, script attempts to store results in MySQL "
             "database using other -db-* parameters.",
    )
    optional.add_argument(
        "-db-pass",
        dest="db_pass",
        default="omniscidb",
        help="Password to use to connect to MySQL database.",
    )
    optional.add_argument(
        "-db-name",
        dest="db_name",
        default="omniscidb",
        help="MySQL database to use to store benchmark results.",
    )
    optional.add_argument(
        "-db-table",
        dest="db_table",
        help="Table to use to store results for this benchmark.",
    )
    # Omnisci server parameters
    required.add_argument(
        "-e",
        "--executable",
        dest="omnisci_executable",
        help="Path to omnisci_server executable.",
    )
    optional.add_argument(
        "-w",
        "--workdir",
        dest="omnisci_cwd",
        help="Path to omnisci working directory. "
             "By default parent directory of executable location is used. "
             "Data directory is used in this location.",
    )
    optional.add_argument(
        "-port",
        "--omnisci_port",
        dest="omnisci_port",
        default=6274,
        type=int,
        help="TCP port number to run omnisci_server on.",
    )
    optional.add_argument(
        "-u",
        "--user",
        dest="user",
        default="admin",
        help="User name to use on omniscidb server.",
    )
    optional.add_argument(
        "-p",
        "--password",
        dest="password",
        default="HyperInteractive",
        help="User password to use on omniscidb server.",
    )
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
    # Additional information
    optional.add_argument(
        "-commit_omnisci",
        dest="commit_omnisci",
        default="1234567890123456789012345678901234567890",
        help="Omnisci commit hash to use for benchmark.",
    )
    optional.add_argument(
        "-commit_ibis",
        dest="commit_ibis",
        default="1234567890123456789012345678901234567890",
        help="Ibis commit hash to use for benchmark.",
    )

    try:
        os.environ["PYTHONIOENCODING"] = "UTF-8"
        os.environ["PYTHONUNBUFFERED"] = "1"

        args = parser.parse_args()

        conda_env = CondaEnvironment(args.env_name)
        if not conda_env.is_env_exist():
            print(f"Conda environment {args.env_name} is not existed.")
            exit(1)

        result_file = f"{args.bench_name}_results.json"
        benchmark_cmd = [
            "python3",
            benchmarks[args.bench_name],
            f"--file={args.file}",
            "--dfiles_num",
            str(args.dfiles_num),
            "--result_file",
            result_file,
            "-no_ml" if args.no_ml else "",
            "-no_ibis" if args.no_ibis else "",
            "-q3_full" if args.q3_full else "",
            f"--optimizer={args.optimizer}" if args.optimizer else "",
            "-pandas_mode",
            args.pandas_mode,
            "-ray_tmpdir",
            args.ray_tmpdir,
            "-ray_memory",
            str(args.ray_memory),
        ]

        omnisci_server_worker = None
        if not args.no_ibis:
            if args.omnisci_executable is None:
                parser.error(
                    "Omnisci executable should be specified with -e/--executable"
                )
            omnisci_server = OmnisciServer(
                omnisci_executable=args.omnisci_executable,
                omnisci_port=args.omnisci_port,
                database_name=args.database_name,
                user=args.user,
                password=args.password,
            )

            omnisci_server_worker = OmnisciServerWorker(omnisci_server)
            pickled_file = "server_worker.pickled"
            cloudpickle.dump(omnisci_server_worker, open(pickled_file, "wb"))
            omnisci_server.launch()

            benchmark_cmd.extend(
                [
                    "--omnisci_server_worker",
                    pickled_file,
                    "--database_name",
                    args.database_name,
                    "--table",
                    args.table,
                    "-dnd" if args.dnd else "",
                    "-dni" if args.dni else "",
                    "-val" if args.validation else "",
                ]
            )

        results = []
        for iter_num in range(1, args.iterations + 1):
            if not args.no_ibis:
                omnisci_server.launch()

            conda_env.run(benchmark_cmd)

            if not args.no_ibis:
                omnisci_server.terminate()

            with open(result_file, "r") as json_file:
                result = json.load(json_file)

            for backend_res in result:
                backend_res["Iteration"] = iter_num
            results.extend(result)

        # if args.db_user is not "":
        #     print("Connecting to database")
        #     db = mysql.connector.connect(
        #         host=args.db_server,
        #         port=args.db_port,
        #         user=args.db_user,
        #         passwd=args.db_pass,
        #         db=args.db_name,
        #     )
        #     db_reporter = DbReport(
        #         db,
        #         args.db_table,
        #         {
        #             "QueryName": "VARCHAR(500) NOT NULL",
        #             "IbisCommitHash": "VARCHAR(500) NOT NULL",
        #             "BackEnd": "VARCHAR(100) NOT NULL",
        #         },
        #         results[0][0]
        #     )
        #     for result in results:
        #           db_reporter.submit(result)

    except Exception:
        traceback.print_exc(file=sys.stdout)
        sys.exit(1)
    finally:
        if omnisci_server:
            omnisci_server.terminate()


if __name__ == "__main__":
    main()
