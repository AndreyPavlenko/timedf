import subprocess
import argparse
import pathlib
import sys
import os

# Load database reporting functions
pathToReportDir = os.path.join(pathlib.Path(__file__).parent, "report")
sys.path.insert(1, pathToReportDir)
import report

parser = argparse.ArgumentParser(description='Run arbitrary omnisci benchmark and submit report values to MySQL database')
optional = parser._action_groups.pop()
required = parser.add_argument_group("required arguments")
parser._action_groups.append(optional)

# Benchmark scripts location
required.add_argument('-path', dest="benchmarks_path", required=True,
                      help="Path to omniscidb/Benchmarks directory.")
# Omnisci server parameters
required.add_argument("-u", "--user", dest="user", default="admin", required=True,
                      help="User name to use on omniscidb server.")
required.add_argument("-p", "--passwd", dest="passwd", default="HyperInteractive", required=True,
                      help="User password to use on omniscidb server.")
required.add_argument("-s", "--server", dest="server", default="localhost", required=True,
                      help="Omniscidb server host name.")
required.add_argument("-o", "--port", dest="port", default="6274", required=True,
                      help="Omniscidb server port number.")
required.add_argument("-n", "--name", dest="name", default="omnisci", required=True,
                      help="Database name to use on omniscidb server.")
required.add_argument("-t", "--import-table-name", dest="import_table_name", required=True,
                      help="Name of table to import data to. NOTE: This table will be dropped before and after the import test, unless --no-drop-table-[before/after] is specified.")
# Required by omnisci benchmark scripts
required.add_argument("-l", "--label", dest="label", required=True,
                      help="Benchmark run label")
required.add_argument("-f", "--import-file", dest="import_file", required=True,
                      help="Absolute path to file or wildcard on omnisci_server machine with data for import test. If wildcard is used, multiple COPY statements are executed to import every file. Number of files may be limited by --max-import-files switch.")
required.add_argument("-c", "--table-schema-file", dest="table_schema_file", required=True,
                      help="Path to local file with CREATE TABLE sql statement for the import table")
required.add_argument("-d", "--queries-dir", dest="queries_dir",
                      help='Absolute path to dir with query files. [Default: "queries" dir in same location as script]')
required.add_argument("-i", "--iterations", dest="iterations", type=int, required=True,
                      help="Number of iterations per query. Must be > 1")

# Fragment size
optional.add_argument('-fs', dest="fragment_size", action='append', type=int,
                      help="Fragment size to use for created table. Multiple values are allowed and encouraged. If no -fs switch is specified, default fragment size is used and templated CREATE TABLE sql files cannot be used.")
optional.add_argument("--max-import-files", dest="max_import_files",
                      help="Maximum number of files to import when -f specifies a wildcard.")

# MySQL database parameters
optional.add_argument("-db-server", default="localhost", help="Host name of MySQL server")
optional.add_argument("-db-port", default=3306, type=int, help="Port number of MySQL server")
optional.add_argument("-db-user", default="", help="Username to use to connect to MySQL database. If user name is specified, script attempts to store results in MySQL database using other -db-* parameters.")
optional.add_argument("-db-pass", default="omniscidb", help="Password to use to connect to MySQL database")
optional.add_argument("-db-name", default="omniscidb", help="MySQL database to use to store benchmark results")

optional.add_argument("-commit", default="1234567890123456789012345678901234567890", help="Commit hash to use to record this benchmark results")

args = parser.parse_args()

import_cmdline = ['python3',
                  os.path.join(args.benchmarks_path, 'run_benchmark_import.py'),
                  '-u', args.user,
                  '-p', args.passwd,
                  '-s', args.server,
                  '-o', args.port,
                  '-n', args.name,
                  '-t', args.import_table_name,
                  '-l', args.label,
                  '-f', args.import_file,
                  '-c', args.table_schema_file,
                  '-e', 'output',
                  '-v',
                  '--no-drop-table-after']

benchmark_cmdline = ['python3',
                     os.path.join(args.benchmarks_path, 'run_benchmark.py'),
                     '-u', args.user,
                     '-p', args.passwd,
                     '-s', args.server,
                     '-o', args.port,
                     '-n', args.name,
                     '-t', args.import_table_name,
                     '-l', args.label,
                     '-d', args.queries_dir,
                     '-i', str(args.iterations),
                     '-e', 'file_json',
                     '-j', 'benchmark.json',
                     '-v']

def execute_process(cmdline):
    try:
        process = subprocess.Popen(cmdline, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        out = process.communicate()[0].strip().decode()
        print(out)
    except OSError as err:
        print("Failed to start", omnisciCmdLine, err)
    if process.returncode != 0:
        print("Command returned", process.returncode)
        sys.exit()

def execute_benchmark(fragment_size):
    cmdline = import_cmdline
    if fragment_size is not None:
        cmdline += ['--fragment_size', str(fragment_size)]
    execute_process(cmdline)

    cmdline = benchmark_cmdline
    execute_process(cmdline)

if args.fragment_size is not None:
    for fs in args.fragment_size:
        print("RUNNING IMPORT WITH FRAGMENT SIZE", fs)
        execute_benchmark(fs)
else:
    print("RUNNING IMPORT WITH DEFAULT FRAGMENT SIZE")
    execute_benchmark(None)

