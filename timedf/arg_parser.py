"""Argument parsing"""
import argparse
from dataclasses import dataclass
from typing import Callable

from .backend import Backend


__all__ = ["add_sql_arguments", "prepare_general_parser", "parse_args", "DbConfig"]


# This can be written as just a function, but we keep the dataclass to add validation and arg parsing in the future.
@dataclass
class DbConfig:
    """Class encapsulates DB configuration and connection.

    For the sqlite you need to pass the path to the file as `name` argument, like
    `name='database.db', driver='sqlite+pysqlite'`.
    """

    driver: str
    server: str = None
    port: int = None
    user: str = None
    password: str = None
    name: str = None

    def is_config_available(self):
        return self.name is not None

    def _validate_driver(self):
        """Provide helpful messages for selected drivers."""
        if self.driver == "mysql+mysqlconnector":
            try:
                import mysql  # noqa: F401
            except ModuleNotFoundError as e:
                raise ModuleNotFoundError(
                    f"Provided DB driver {self.driver}, but it is not installed.\n"
                    "You can install it with `pip install mysql-connector-python`\n"
                ) from e

    def _create_engine(self):
        from sqlalchemy import create_engine
        from sqlalchemy.engine.url import URL
        from sqlalchemy.pool import NullPool

        self._validate_driver()

        url = URL.create(
            host=self.server,
            drivername=self.driver,
            username=self.user,
            password=self.password,
            port=self.port,
            database=self.name,
        )

        # After moving from mySQL to MariaDB long-running benchmarks (~2hrs) had error during DB writing
        # sqlalchemy.exc.OperationalError: (mysql.connector.errors.OperationalError) MySQL Connection not available
        # This is likely caused by closed connection in the pool. We avoid using connection pool
        # to solve this problem. This will make each write a bit slower because now we create
        # connection for each request, but since benchmark run creates very few writes that shouldn't be a problem
        return create_engine(url, poolclass=NullPool)

    def maybeCreateBenchmarkDb(self):
        if self.is_config_available():
            from .report import BenchmarkDb

            return BenchmarkDb(self._create_engine())
        else:
            return None


def add_sql_arguments(parser):
    parser.add_argument(
        "-db_driver",
        dest="db_driver",
        default="sqlite+pysqlite",
        help="Driver for the sql table in sqlalchemy format.",
    )
    parser.add_argument("-db_server", dest="db_server", help="Host name of SQL server.")
    parser.add_argument("-db_port", dest="db_port", type=int, help="Port number of SQL server.")
    parser.add_argument(
        "-db_user",
        dest="db_user",
        help="Username to use to connect to SQL database. "
        "If user name is specified, script attempts to store results in SQL "
        "database using other -db-* parameters.",
    )
    parser.add_argument(
        "-db_pass",
        dest="db_pass",
        help="Password to use to connect to SQL database.",
    )
    parser.add_argument(
        "-db_name",
        dest="db_name",
        help="SQL database to use to store benchmark results.",
    )


def prepare_general_parser():
    parser = argparse.ArgumentParser(description="Run benchmarks for Modin perf testing")
    benchmark = parser.add_argument_group("benchmark")
    sql = parser.add_argument_group("sql")
    commits = parser.add_argument_group("commits")

    # Benchmark parameters
    benchmark.add_argument("bench_name", help="Benchmark to run.")
    benchmark.add_argument("-data_file", help="A datafile that should be loaded.", required=True)
    benchmark.add_argument(
        "-iterations",
        default=1,
        type=int,
        help="Number of iterations to run. All results will be submitted to the DB.",
    )
    benchmark.add_argument(
        "-backend",
        "-pandas_mode",
        choices=Backend.supported_backends,
        default="Pandas",
        help="Specifies which backend to use: "
        "plain Pandas, Modin runing on Ray or on Dask or on HDK",
    )
    benchmark.add_argument(
        "-ray_tmpdir",
        default="./tmp",
        help="Location where to keep Ray plasma store. "
        "It should have enough space to keep -ray_memory",
    )
    benchmark.add_argument(
        "-ray_memory",
        default=200 * 1024 * 1024 * 1024,
        type=int,
        help="Size of memory to allocate for Ray plasma store",
    )
    benchmark.add_argument(
        "-verbosity",
        help="""Level of verbosity for timers. Use 1, 2 or 3 if you want to get more logging info.
        Level 0: no writing (default)
              1: write about exit only
              2: write about exit and enter
              3: write profiler statistics""",
        default=0,
        type=int,
        choices=(0, 1, 2, 3),
    )
    benchmark.add_argument(
        "-no_ml",
        default=False,
        action="store_true",
        help="Do not run machine learning benchmark, only ETL part",
    )
    benchmark.add_argument(
        "-use_modin_xgb",
        default=False,
        action="store_true",
        help="Whether to use Modin XGBoost for ML part, relevant for `plasticc` and `ny_taxi_ml` "
        " benchmark.",
    )
    benchmark.add_argument(
        "-save_benchmark_name",
        "-save_name",
        default=None,
        help="Save benchmark in DB under this name. Saves with `bench_name` name by default.",
    )

    benchmark.add_argument(
        "-save_backend_name",
        default=None,
        help="Save backend in DB under this name. Saves with `backend` name by default.",
    )
    benchmark.add_argument(
        "-tag",
        default=None,
        help="Tag this run with provided string to be able to find it in the database. Useful when user configure backend with some paramenters and want to be able to find results later on.",
    )
    # SQL database parameters
    add_sql_arguments(sql)
    # Additional information
    commits.add_argument(
        "-commit_hdk",
        default="1234567890123456789012345678901234567890",
        help="HDK commit hash used for tests.",
    )
    commits.add_argument(
        "-commit_timedf",
        default="1234567890123456789012345678901234567890",
        help="timedf commit hash used for tests.",
    )
    commits.add_argument(
        "-commit_modin",
        default="1234567890123456789012345678901234567890",
        help="Modin commit hash used for tests.",
    )
    commits.add_argument(
        "-num_threads",
        default=None,
        type=int,
        help="Number of threads used for data processing",
    )
    return parser


def parse_args(add_benchmark_args: Callable[[argparse.ArgumentParser], None]):
    """Parse arguments including benchmark-specific arguments that will be added using provided
    `add_benchmark_args` callable"""
    parser = prepare_general_parser()
    benchmark_parser = parser.add_argument_group("benchmark_specific")
    add_benchmark_args(benchmark_parser)
    args = parser.parse_args()

    db_config = DbConfig(
        driver=args.db_driver,
        server=args.db_server,
        port=args.db_port,
        user=args.db_user,
        password=args.db_pass,
        name=args.db_name,
    )

    return args, db_config
