import os
import sys
import argparse
from server import OmnisciServer
from server import execute_process

sys.path.append(os.path.dirname(__file__))


def str_arg_to_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Cannot recognize boolean value.')


def add_conda_execution(cmdline):
    cmd_res = ['conda', 'run', '-n', args.env_name]
    cmd_res.extend(cmdline)
    return cmd_res


def combinate_requirements(ibis, ci, res):
    with open(res, "w") as f_res:
        with open(ibis) as f_ibis:
            for line in f_ibis:
                f_res.write(line)
        with open(ci) as f_ci:
            for line in f_ci:
                f_res.write(line) 


omniscript_path = os.path.dirname(__file__)

parser = argparse.ArgumentParser(description='Run internal tests from ibis project')
optional = parser._action_groups.pop()
required = parser.add_argument_group("required arguments")
parser._action_groups.append(optional)

possible_tasks = ['build', 'test', 'benchmark']
benchmarks = {'ny_taxi': os.path.join(omniscript_path, "taxi", "taxibench_ibis.py")}
# Task
required.add_argument("-t", "--task", dest="task", required=True,
                      help=f"Task for execute {possible_tasks}. Use , separator for multiple tasks")

# Environment
required.add_argument('-en', '--env_name', dest="env_name", default="ibis-tests",
                      help="Conda env name.")
optional.add_argument('-ec', '--env_check', dest="env_check", default=False, type=str_arg_to_bool,
                      help="Check if env exists. If it exists don't recreate.")
optional.add_argument('-s', '--save_env', dest="save_env", default=False, type=str_arg_to_bool,
                      help="Save conda env after executing.")
optional.add_argument('-r', '--report_path', dest="report_path",
                      default=os.path.join(omniscript_path, ".."), help="Path to report file.")
optional.add_argument('-ci', '--ci_requirements', dest="ci_requirements",
                      default=os.path.join(omniscript_path, "ci_requirements.yml"),
                      help="File with ci requirements for conda env.")
optional.add_argument('-py', '--python_version', dest="python_version", default="3.7",
                      help="File with ci requirements for conda env.")
# Ibis
required.add_argument('-i', '--ibis_path', dest="ibis_path", required=True,
                      help="Path to ibis directory.")
# Benchmarks
optional.add_argument('-bn', '--bench_name', dest="bench_name",
                      help=f"Benchmark name. Supports {list(benchmarks.keys())}")
# Omnisci server parameters
optional.add_argument("-e", "--executable", dest="omnisci_executable", required=True,
                      help="Path to omnisci_server executable.")
optional.add_argument("-w", "--workdir", dest="omnisci_cwd",
                      help="Path to omnisci working directory. "
                           "By default parent directory of executable location is used. "
                           "Data directory is used in this location.")
optional.add_argument("-o", "--l", dest="omnisci_port", default=6274, type=int,
                      help="TCP port number to run omnisci_server on.")
optional.add_argument("-u", "--user", dest="user", default="admin",
                      help="User name to use on omniscidb server.")
optional.add_argument("-p", "--password", dest="password", default="HyperInteractive",
                      help="User password to use on omniscidb server.")
optional.add_argument("-n", "--name", dest="name", default="agent_test_ibis", required=True,
                      help="Database name to use on omniscidb server.")

optional.add_argument("-commit_omnisci", dest="commit_omnisci",
                      default="1234567890123456789012345678901234567890",
                      help="Omnisci commit hash to use for tests.")
optional.add_argument("-commit_ibis", dest="commit_ibis",
                      default="1234567890123456789012345678901234567890",
                      help="Ibis commit hash to use for tests.")


args = parser.parse_args()

os.environ["IBIS_TEST_OMNISCIDB_DATABASE"] = args.name
os.environ["IBIS_TEST_DATA_DB"] = args.name

required_tasks = args.task.split(',')
tasks = {}
task_checker = False
for task in possible_tasks:
    if task in required_tasks:
        tasks[task] = True
        task_checker = True
    else:
        tasks[task] = False
if not task_checker:
    print(f"Only {list(tasks.keys())} are supported, {required_tasks} cannot find possible tasks")
    sys.exit(1)

if args.python_version not in ['3.7', '3,6']:
    print(f"Only 3.7 and 3.6 python versions are supported, {args.python_version} is not supported")
    sys.exit(1)
ibis_requirements = os.path.join(args.ibis_path, "ci",
                                 f"requirements-{args.python_version}-dev.yml")
ibis_data_script = os.path.join(args.ibis_path, "ci", "datamgr.py")

requirements_file = "requirements.yml"
report_file_name = f"report-{args.commit_ibis[:8]}-{args.commit_omnisci[:8]}.html"
if not os.path.isdir(args.report_path):
    os.makedirs(args.report_path)
report_file_path = os.path.join(args.report_path, report_file_name)

install_ibis_cmdline = ['python3',
                os.path.join('setup.py'),
                'install',
                '--user']

check_env_cmdline = ['conda',
                     'env',
                     'list']

create_env_cmdline = ['conda',
                      'env',
                      'create',
                      '--name', args.env_name,
                      '--file', requirements_file]

remove_env_cmdline = ['conda',
                      'env',
                      'remove',
                      '--name', args.env_name]

dataset_download_cmdline = ['python3',
                            ibis_data_script,
                            'download']

dataset_import_cmdline = ['python3',
                          ibis_data_script,
                          'omniscidb',
                          '-P', str(args.omnisci_port),
                          '--database', args.name]

ibis_tests_cmdline = ['pytest',
                      '-m', 'omniscidb',
                      '--disable-pytest-warnings',
                      f'--html={report_file_path}']

if tasks['benchmark']:
    if not args.bench_name or args.bench_name not in benchmarks.keys():
        print(f"Benchmark {args.bench_name} is not supported, /only {list(benchmarks.keys())} are supported")
        sys.exit(1)

    datafiles = "'/localdisk/benchmark_datasets/taxi/trips_xa{a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t}.csv.gz'"
    ny_taxi_bench_cmdline = ['python3',
                    benchmarks[args.bench_name],
                    '-e', args.omnisci_executable,
                    '-port', str(args.omnisci_port),
                    '-db-port', '3306',
                    '-df', '20',
                    '-dp', datafiles,
                    '-i', '5',
                    '-u', args.user,
                    '-p', args.password,
                    '-db-server=ansatlin07.an.intel.com',
                    '-n', args.name,
                    f'-db-user=gashiman',
                    f'-db-pass=omniscidb',
                    f'-db-name=omniscidb',
                    '-db-table=taxibench_ibis',
                    '-commit_omnisci', args.commit_omnisci,
                    '-commit_ibis', args.commit_ibis]

    benchmarks_cmd = {'ny_taxi': ny_taxi_bench_cmdline}

try:
    omnisci_server = None
    print("PREPARING ENVIRONMENT")
    combinate_requirements(ibis_requirements, args.ci_requirements, requirements_file)
    _, envs = execute_process(check_env_cmdline)
    if args.env_name in envs:
        if args.env_check is False:
            execute_process(remove_env_cmdline)
            execute_process(create_env_cmdline, print_output=False)
    else:
        execute_process(create_env_cmdline, print_output=False)

    if tasks['build']:
        print("IBIS INSTALLATION")
        execute_process(add_conda_execution(install_ibis_cmdline), cwd=args.ibis_path,
                        print_output=False)

    if tasks['test']:
        print("STARTING OMNISCI SERVER")
        omnisci_server = OmnisciServer(omnisci_executable=args.omnisci_executable,
                                       omnisci_port=args.omnisci_port,database_name=args.name,
                                       omnisci_cwd=args.omnisci_cwd, user=args.user,
                                       password=args.password)
        omnisci_server.launch()

    if tasks['test']:
        print("PREPARING DATA")
        execute_process(add_conda_execution(dataset_download_cmdline))
        execute_process(add_conda_execution(dataset_import_cmdline))

        print("RUNNING TESTS")
        execute_process(add_conda_execution(ibis_tests_cmdline), cwd=args.ibis_path)
    
    if tasks['benchmark']:
        print(f"RUNNING BENCHMARK {args.bench_name}")
        execute_process(add_conda_execution(benchmarks_cmd[args.bench_name]))
    
except Exception as err:
    print("Failed", err)
    sys.exit(1)

finally:
    if omnisci_server:
        omnisci_server.terminate()
    if args.save_env is False:
        execute_process(remove_env_cmdline)
