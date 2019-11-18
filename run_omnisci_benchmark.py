from braceexpand import braceexpand
import mysql.connector
import subprocess
import threading
import argparse
import pathlib
import signal
import glob
import time
import json
import sys
import os
import io

def execute_process(cmdline):
    try:
        process = subprocess.Popen(cmdline, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        out = process.communicate()[0].strip().decode()
        print(out)
    except OSError as err:
        print("Failed to start", omnisciCmdLine, err)
    if process.returncode != 0:
        print("Command returned", process.returncode)
        sys.exit(2)

def execute_benchmark(fragment_size, report):
    cmdline = import_cmdline
    if fragment_size is not None:
        cmdline += ['--fragment-size', str(fragment_size)]
        fs = fragment_size
    else:
        fs = 0
    execute_process(cmdline)

    cmdline = benchmark_cmdline
    execute_process(cmdline)
    with open("benchmark.json", "r") as results_file:
        results = json.load(results_file)
    for result in results:
        print(datafiles, ",",
              fs, ",",
              result['name'], ",",
              result['results']['query_exec_avg'], ",",
              result['results']['query_total_avg'], ",",
              result['results']['query_error_info'],
              '\n', file=report, sep='', end='', flush=True)
        if db_reporter is not None:
            db_reporter.submit({
                'FilesNumber': datafiles,
                'FragmentSize': fs,
                'BenchName': result['name'],
                'BestExecTimeMS': str(result['results']['query_exec_avg']),
                'BestTotalTimeMS': result['results']['query_total_avg']
            })

def print_omnisci_output(stdout):
    for line in iter(stdout.readline, b''):
        print("OMNISCI>>", line.decode().strip())

# Load database reporting functions
pathToReportDir = os.path.join(pathlib.Path(__file__).parent, "report")
sys.path.insert(1, pathToReportDir)
import report

parser = argparse.ArgumentParser(description='Run arbitrary omnisci benchmark and submit report values to MySQL database')
optional = parser._action_groups.pop()
required = parser.add_argument_group("required arguments")
parser._action_groups.append(optional)

# Benchmark scripts location
required.add_argument('-r', '--report', dest="report", default="report.csv",
                    help="Report file name")
required.add_argument('-path', dest="benchmarks_path", required=True,
                      help="Path to omniscidb/Benchmarks directory.")
# Omnisci server parameters
required.add_argument("-e", "--executable", dest="omnisci_executable", required=True,
                      help="Path to omnisci_server executable.")
optional.add_argument("-w", "--workdir", dest="omnisci_cwd",
                      help="Path to omnisci working directory. By default parent directory of executable location is used.")
optional.add_argument("-o", "--port", dest="omnisci_port", default=62274, type=int,
                      help="TCP port number to run omnisci_server on.")
required.add_argument("-u", "--user", dest="user", default="admin", required=True,
                      help="User name to use on omniscidb server.")
required.add_argument("-p", "--passwd", dest="passwd", default="HyperInteractive", required=True,
                      help="User password to use on omniscidb server.")
required.add_argument("-n", "--name", dest="name", default="omnisci", required=True,
                      help="Database name to use on omniscidb server.")
required.add_argument("-t", "--import-table-name", dest="import_table_name", required=True,
                      help="Name of table to import data to. NOTE: This table will be dropped before and after the import test, unless --no-drop-table-[before/after] is specified.")
# Required by omnisci benchmark scripts
required.add_argument("-l", "--label", dest="label", required=True,
                      help="Benchmark run label")
required.add_argument("-f", "--import-file", dest="import_file", required=True,
                      help="Absolute path to file or wildcard on omnisci_server machine with data for import test. If wildcard is used, all files are imported in one COPY statement. Limiting number of files is possible using curly braces wildcard, e.g. trips_xa{a,b,c}.csv.gz.")
required.add_argument("-c", "--table-schema-file", dest="table_schema_file", required=True,
                      help="Path to local file with CREATE TABLE sql statement for the import table")
required.add_argument("-d", "--queries-dir", dest="queries_dir",
                      help='Absolute path to dir with query files. [Default: "queries" dir in same location as script]')
required.add_argument("-i", "--iterations", dest="iterations", type=int, required=True,
                      help="Number of iterations per query. Must be > 1")

# Fragment size
optional.add_argument('-fs', dest="fragment_size", action='append', type=int,
                      help="Fragment size to use for created table. Multiple values are allowed and encouraged. If no -fs switch is specified, default fragment size is used and templated CREATE TABLE sql files cannot be used.")

# MySQL database parameters
optional.add_argument("-db-server", default="localhost", help="Host name of MySQL server")
optional.add_argument("-db-port", default=3306, type=int, help="Port number of MySQL server")
optional.add_argument("-db-user", default="", help="Username to use to connect to MySQL database. If user name is specified, script attempts to store results in MySQL database using other -db-* parameters.")
optional.add_argument("-db-pass", default="omniscidb", help="Password to use to connect to MySQL database")
optional.add_argument("-db-name", default="omniscidb", help="MySQL database to use to store benchmark results")

optional.add_argument("-commit", default="1234567890123456789012345678901234567890", help="Commit hash to use to record this benchmark results")

args = parser.parse_args()

server_cmdline = [args.omnisci_executable,
                  'data',
                  '--port', str(args.omnisci_port),
                  '--http-port', "62278",
                  '--calcite-port', "62279",
                  '--config', 'omnisci.conf']

import_cmdline = ['python3',
                  os.path.join(args.benchmarks_path, 'run_benchmark_import.py'),
                  '-u', args.user,
                  '-p', args.passwd,
                  '-s', 'localhost',
                  '-o', str(args.omnisci_port),
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
                     '-s', 'localhost',
                     '-o', str(args.omnisci_port),
                     '-n', args.name,
                     '-t', args.import_table_name,
                     '-l', args.label,
                     '-d', args.queries_dir,
                     '-i', str(args.iterations),
                     '-e', 'file_json',
                     '-j', 'benchmark.json',
                     '-v']

db_reporter = None
if args.db_user is not "":
    print("Connecting to database")
    db = mysql.connector.connect(host=args.db_server, port=args.db_port, user=args.db_user, passwd=args.db_pass, db=args.db_name);
    db_reporter = report.DbReport(db, "taxibench", {
        'FilesNumber': 'INT UNSIGNED NOT NULL',
        'FragmentSize': 'BIGINT UNSIGNED NOT NULL',
        'BenchName': 'VARCHAR(500) NOT NULL',
        'BestExecTimeMS': 'BIGINT UNSIGNED',
        'BestTotalTimeMS': 'BIGINT UNSIGNED'
    }, {
        'ScriptName': 'taxibench.py',
        'CommitHash': args.commit
    })


# Use bash to determine number of matching files because python
# doesn't support curly brace expansion
#cmdline="bash -c 'ls -1 " + args.import_file + " | wc -l '"
#datafiles = int(subprocess.check_output(cmdline, shell=True).decode().strip())

datafiles = list(braceexpand(args.import_file))
datafiles = [x for f in datafiles for x in glob.glob(f)]
print("NUMBER OF DATAFILES FOUND:", len(datafiles))

if args.omnisci_cwd is not None:
    server_cwd = args.omnisci_cwd
else:
    server_cwd = pathlib.Path(args.omnisci_executable).parent.parent
try:
    server_process = subprocess.Popen(server_cmdline, cwd=server_cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
except OSError as err:
    print("Failed to start", omnisciCmdLine, err)
    sys.exit(1)
pt = threading.Thread(target=print_omnisci_output, args=(server_process.stdout,), daemon=True)
pt.start()

# Allow server to start up. It has to open TCP port and start
# listening, otherwise the following benchmarks fail.
time.sleep(5)

with open(args.report, "w") as report:
    if args.fragment_size is not None:
        for fs in args.fragment_size:
            print("RUNNING IMPORT WITH FRAGMENT SIZE", fs)
            execute_benchmark(fs, report)
    else:
        print("RUNNING IMPORT WITH DEFAULT FRAGMENT SIZE")
        execute_benchmark(None, report)

print("TERMINATING SERVER")
server_process.send_signal(signal.SIGINT)
time.sleep(1)
server_process.kill()
time.sleep(1)
server_process.terminate()
