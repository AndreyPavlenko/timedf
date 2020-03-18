import argparse
import os
import sys
import traceback

from environment import CondaEnvironment
from server import OmnisciServer
from utils import combinate_requirements, str_arg_to_bool


def main():
    omniscript_path = os.path.dirname(__file__)
    omnisci_server = None
    args = None

    parser = argparse.ArgumentParser(description="Run internal tests from ibis project")
    optional = parser._action_groups.pop()
    required = parser.add_argument_group("required arguments")
    parser._action_groups.append(optional)

    possible_tasks = ["build", "test"]
    benchmarks = ["ny_taxi", "santander", "census", "plasticc"]
    # Task
    required.add_argument(
        "-t",
        "--task",
        dest="task",
        required=True,
        help=f"Task for execute {possible_tasks}. Use , separator for multiple tasks",
    )

    # Environment
    required.add_argument("-en", "--env_name", dest="env_name", help="Conda env name.")
    optional.add_argument(
        "-ec",
        "--env_check",
        dest="env_check",
        default=False,
        type=str_arg_to_bool,
        help="Check if env exists. If it exists don't recreate.",
    )
    optional.add_argument(
        "-s",
        "--save_env",
        dest="save_env",
        default=False,
        type=str_arg_to_bool,
        help="Save conda env after executing.",
    )
    optional.add_argument(
        "-r",
        "--report_path",
        dest="report_path",
        default=os.path.join(omniscript_path, ".."),
        help="Path to report file.",
    )
    optional.add_argument(
        "-ci",
        "--ci_requirements",
        dest="ci_requirements",
        default=os.path.join(omniscript_path, "ci_requirements.yml"),
        help="File with ci requirements for conda env.",
    )
    optional.add_argument(
        "-py",
        "--python_version",
        dest="python_version",
        default="3.7",
        help="File with ci requirements for conda env.",
    )
    # Ibis
    required.add_argument(
        "-i",
        "--ibis_path",
        dest="ibis_path",
        required=True,
        help="Path to ibis directory.",
    )
    # Ibis tests
    optional.add_argument(
        "--expression",
        dest="expression",
        default=" ",
        help="Run tests which match the given substring test names and their parent "
        "classes. Example: 'test_other', while 'not test_method' matches those "
        "that don't contain 'test_method' in their names.",
    )
    # Omnisci server parameters
    optional.add_argument(
        "-e",
        "--executable",
        dest="omnisci_executable",
        required=True,
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
        default="agent_test_ibis",
        help="Database name to use in omniscidb server.",
    )

    optional.add_argument(
        "-commit_omnisci",
        dest="commit_omnisci",
        default="1234567890123456789012345678901234567890",
        help="Omnisci commit hash to use for tests.",
    )
    optional.add_argument(
        "-commit_ibis",
        dest="commit_ibis",
        default="1234567890123456789012345678901234567890",
        help="Ibis commit hash to use for tests.",
    )

    try:
        args = parser.parse_args()

        os.environ["IBIS_TEST_OMNISCIDB_DATABASE"] = args.database_name
        os.environ["IBIS_TEST_DATA_DB"] = args.database_name
        os.environ["IBIS_TEST_OMNISCIDB_PORT"] = str(args.omnisci_port)
        os.environ["PYTHONIOENCODING"] = "UTF-8"
        os.environ["PYTHONUNBUFFERED"] = "1"

        required_tasks = args.task.split(",")
        tasks = {}
        for task in possible_tasks:
            tasks[task] = True if task in required_tasks else False

        if True not in list(tasks.values()):
            print(
                f"Only {list(tasks.keys())} are supported, {required_tasks} cannot find possible tasks"
            )
            sys.exit(1)

        if args.python_version not in ["3.7", "3,6"]:
            print(
                f"Only 3.7 and 3.6 python versions are supported, {args.python_version} is not supported"
            )
            sys.exit(1)
        ibis_requirements = os.path.join(
            args.ibis_path, "ci", f"requirements-{args.python_version}-dev.yml"
        )
        ibis_data_script = os.path.join(args.ibis_path, "ci", "datamgr.py")

        requirements_file = "requirements.yml"
        report_file_name = (
            f"report-{args.commit_ibis[:8]}-{args.commit_omnisci[:8]}.html"
        )
        if not os.path.isdir(args.report_path):
            os.makedirs(args.report_path)
        report_file_path = os.path.join(args.report_path, report_file_name)

        install_ibis_cmdline = ["python3", os.path.join("setup.py"), "install"]

        dataset_download_cmdline = ["python3", ibis_data_script, "download"]

        dataset_import_cmdline = [
            "python3",
            ibis_data_script,
            "omniscidb",
            "-P",
            str(args.omnisci_port),
            "--database",
            args.database_name,
        ]

        ibis_tests_cmdline = [
            "pytest",
            "-m",
            "omniscidb",
            "--disable-pytest-warnings",
            "-k",
            args.expression,
            f"--html={report_file_path}",
        ]

        conda_env = CondaEnvironment(args.env_name)

        print("PREPARING ENVIRONMENT")
        combinate_requirements(
            ibis_requirements, args.ci_requirements, requirements_file
        )
        conda_env.create(args.env_check, requirements_file=requirements_file)

        if tasks["build"]:
            print("IBIS INSTALLATION")
            conda_env.run(install_ibis_cmdline, cwd=args.ibis_path, print_output=False)

        if tasks["test"]:
            print("STARTING OMNISCI SERVER")
            omnisci_server = OmnisciServer(
                omnisci_executable=args.omnisci_executable,
                omnisci_port=args.omnisci_port,
                database_name=args.database_name,
                omnisci_cwd=args.omnisci_cwd,
                user=args.user,
                password=args.password,
            )
            omnisci_server.launch()

            print("PREPARING DATA")
            conda_env.run(dataset_download_cmdline)
            conda_env.run(dataset_import_cmdline)

            print("RUNNING TESTS")
            conda_env.run(ibis_tests_cmdline, cwd=args.ibis_path)

    except Exception:
        traceback.print_exc(file=sys.stdout)
        sys.exit(1)

    finally:
        if omnisci_server:
            omnisci_server.terminate()
        if args and args.save_env is False:
            conda_env.remove()


if __name__ == "__main__":
    main()
