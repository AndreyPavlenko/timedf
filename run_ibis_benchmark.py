# coding: utf-8
import argparse
import os
import sys
import traceback

import mysql.connector

from report import DbReport
from server import OmnisciServer
from server_worker import OmnisciServerWorker
from utils import random_if_default, str_arg_to_bool


def main():
    omniscript_path = os.path.dirname(__file__)
    args = None
    omnisci_server = None

    benchmarks = ["ny_taxi", "santander", "census", "plasticc"]

    parser = argparse.ArgumentParser(description="Run internal tests from ibis project")
    optional = parser._action_groups.pop()
    required = parser.add_argument_group("required arguments")
    parser._action_groups.append(optional)

    required.add_argument(
        "-bench_name", dest="bench_name", choices=benchmarks, help="Benchmark name.",
    )
    required.add_argument(
        "-data_file", dest="data_file", help="A datafile that should be loaded.",
    )
    optional.add_argument(
        "-dfiles_num",
        dest="dfiles_num",
        default=1,
        type=int,
        help="Number of datafiles to input into database for processing.",
    )
    optional.add_argument(
        "-iterations",
        dest="iterations",
        default=5,
        type=int,
        help="Number of iterations to run every query. Best result is selected.",
    )
    optional.add_argument(
        "-dnd", default=False, type=str_arg_to_bool, help="Do not delete old table."
    )
    optional.add_argument(
        "-dni",
        default=False,
        type=str_arg_to_bool,
        help="Do not create new table and import any data from CSV files.",
    )
    optional.add_argument(
        "-validation",
        dest="validation",
        default=False,
        type=str_arg_to_bool,
        help="validate queries results (by comparison with Pandas queries results).",
    )
    optional.add_argument(
        "-optimizer",
        choices=["intel", "stock"],
        dest="optimizer",
        default="intel",
        help="Which optimizer is used",
    )
    optional.add_argument(
        "-no_ibis",
        default=False,
        type=str_arg_to_bool,
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
        default=False,
        type=str_arg_to_bool,
        help="Do not run machine learning benchmark, only ETL part",
    )
    optional.add_argument(
        "-gpu-memory",
        dest="gpu_memory",
        type=int,
        help="specify the memory of your gpu, default 16. (This controls the lines to be used. Also work for CPU version. )",
        default=16,
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
    optional.add_argument(
        "-executable", dest="executable", help="Path to omnisci_server executable.",
    )
    optional.add_argument(
        "-omnisci_cwd",
        dest="omnisci_cwd",
        help="Path to omnisci working directory. "
        "By default parent directory of executable location is used. "
        "Data directory is used in this location.",
    )
    optional.add_argument(
        "-port",
        dest="port",
        default=-1,
        type=int,
        help="TCP port number to run omnisci_server on.",
    )
    optional.add_argument(
        "-http-port",
        dest="http_port",
        default=-1,
        type=int,
        help="HTTP port number to run omnisci_server on.",
    )
    optional.add_argument(
        "-calcite-port",
        dest="calcite_port",
        default=-1,
        type=int,
        help="Calcite port number to run omnisci_server on.",
    )
    optional.add_argument(
        "-user",
        dest="user",
        default="admin",
        help="User name to use on omniscidb server.",
    )
    optional.add_argument(
        "-password",
        dest="password",
        default="HyperInteractive",
        help="User password to use on omniscidb server.",
    )
    optional.add_argument(
        "-database_name",
        dest="database_name",
        default="omnisci",
        help="Database name to use in omniscidb server.",
    )
    optional.add_argument(
        "-table",
        dest="table",
        default="benchmark_table",
        help="Table name name to use in omniscidb server.",
    )
    optional.add_argument(
        "-ipc_conn",
        dest="ipc_connection",
        default=True,
        type=str_arg_to_bool,
        help="Table name name to use in omniscidb server.",
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

        args.port = random_if_default(
            value=args.port, least=60000, greater=69999, default=-1
        )
        args.http_port = random_if_default(
            value=args.http_port, least=60000, greater=69999, default=-1
        )
        args.calcite_port = random_if_default(
            value=args.calcite_port, least=60000, greater=69999, default=-1
        )

        if args.bench_name == "ny_taxi":
            from taxi import run_benchmark
        elif args.bench_name == "santander":
            from santander import run_benchmark
        elif args.bench_name == "census":
            from census import run_benchmark
        elif args.bench_name == "plasticc":
            from plasticc import run_benchmark

        parameters = {
            "data_file": args.data_file,
            "dfiles_num": args.dfiles_num,
            "no_ml": args.no_ml,
            "no_ibis": args.no_ibis,
            "optimizer": args.optimizer,
            "pandas_mode": args.pandas_mode,
            "ray_tmpdir": args.ray_tmpdir,
            "ray_memory": args.ray_memory,
            "gpu_memory": args.gpu_memory,
        }

        omnisci_server_worker = None
        if not args.no_ibis:
            if args.executable is None:
                parser.error(
                    "Omnisci executable should be specified with -e/--executable for Ibis part"
                )
            omnisci_server = OmnisciServer(
                omnisci_executable=args.executable,
                omnisci_port=args.port,
                http_port=args.http_port,
                calcite_port=args.calcite_port,
                database_name=args.database_name,
                user=args.user,
                password=args.password,
            )

            parameters["database_name"] = args.database_name
            parameters["table"] = args.table
            parameters["dnd"] = args.dnd
            parameters["dni"] = args.dni
            parameters["validation"] = args.validation

        etl_results = []
        ml_results = []
        print(parameters)
        for iter_num in range(1, args.iterations + 1):
            print(f"Iteration №{iter_num}")

            if not args.no_ibis:
                omnisci_server_worker = OmnisciServerWorker(omnisci_server)
                parameters["omnisci_server_worker"] = omnisci_server_worker
                parameters["connect_to_sever"] = \
                    omnisci_server_worker.ipc_connect_to_server() if args.ipc_connection else omnisci_server_worker.connect_to_server()
                omnisci_server.launch()

            result = run_benchmark(parameters)

            if not args.no_ibis:
                omnisci_server.terminate()

            for backend_res in result["ETL"]:
                if backend_res:
                    backend_res["Iteration"] = iter_num
                    etl_results.append(backend_res)
            for backend_res in result["ML"]:
                if backend_res:
                    backend_res["Iteration"] = iter_num
                    ml_results.append(backend_res)

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
        omnisci_server = None
    except Exception:
        traceback.print_exc(file=sys.stdout)
        sys.exit(1)
    finally:
        if omnisci_server:
            omnisci_server.terminate()


if __name__ == "__main__":
    main()
