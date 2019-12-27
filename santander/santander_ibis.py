import mysql.connector
import pandas as pd
import numpy as np
import subprocess
import argparse
import pathlib
import time
import glob
import sys
import os

omnisci_executable  = "build/bin/omnisql"
datafile_directory = "/localdisk/work/train.csv"
database_name = "santanderdb"
table_name = "train_table"

# Load database reporting, server and Ibis modules
path_to_report_dir = os.path.join(pathlib.Path(__file__).parent, "..", "report")
print(path_to_report_dir)
path_to_server_dir = os.path.join(pathlib.Path(__file__).parent, "..", "server")
path_to_ibis_dir = os.path.join(pathlib.Path(__file__).parent.parent, "..", "ibis/build/lib")
sys.path.insert(1, path_to_report_dir)
sys.path.insert(1, path_to_server_dir)
sys.path.insert(1, path_to_ibis_dir)
import report
import server2
import ibis

parser = argparse.ArgumentParser(description='Run Santander benchmark using Ibis.')

parser.add_argument('-e', default=omnisci_executable, help='Path to executable "omnisql".')
parser.add_argument('-r', default="report_santander_ibis.csv", help="Report file name.")
parser.add_argument('-dp', default=datafile_directory, help="Datafile that should be loaded.")
parser.add_argument('-i', default=5, type=int, help="Number of iterations to run every query. Best result is selected.")
parser.add_argument('-dnd', action='store_true', help="Do not delete old table.")
parser.add_argument('-dni', action='store_true', help="Do not create new table and import any data from CSV files.")
parser.add_argument("-port", default=62074, type=int, help="TCP port that omnisql client should use to connect to server.")

parser.add_argument("-db-server", default="localhost", help="Host name of MySQL server.")
parser.add_argument("-db-port", default=3306, type=int, help="Port number of MySQL server.")
parser.add_argument("-db-user", default="", help="Username to use to connect to MySQL database. If user name is specified, script attempts to store results in MySQL database using other -db-* parameters.")
parser.add_argument("-db-pass", default="omniscidb", help="Password to use to connect to MySQL database.")
parser.add_argument("-db-name", default="omniscidb", help="MySQL database to use to store benchmark results.")
parser.add_argument("-db-table", help="Table to use to store results for this benchmark.")

parser.add_argument("-commit", default="1234567890123456789012345678901234567890", help="Commit hash to use to record this benchmark results.")

args = parser.parse_args()

if args.i < 1:
    print("Bad number of iterations specified", args.i)

omnisci_server = server2.Omnisci_server(omnisci_executable=args.e, omnisci_port=args.port, database_name=database_name, table_name=table_name)
omnisci_server.launch()

time.sleep(2)
conn = omnisci_server.connect_to_server()

datafile_columns_names = ["ID_code", "target"] + ["var_" + str(index) for index in range(200)]
datafile_columns_types = ["string", "Boolean"] + ["float32" for _ in range(200)]

schema_train = ibis.Schema(
    names_train = datafile_columns_names,
    types_train = datafile_columns_types
)

'''
db_reporter = None
if args.db_user is not "":
    print("Connecting to database")
    db = mysql.connector.connect(host=args.db_server, port=args.db_port, user=args.db_user, passwd=args.db_pass, db=args.db_name)
    db_reporter = report.DbReport(db, args.db_table, {
        'FilesNumber': 'INT UNSIGNED NOT NULL',
        'QueryName': 'VARCHAR(500) NOT NULL',
        'FirstExecTimeMS': 'BIGINT UNSIGNED',
        'WorstExecTimeMS': 'BIGINT UNSIGNED',
        'BestExecTimeMS': 'BIGINT UNSIGNED',
        'AverageExecTimeMS': 'BIGINT UNSIGNED',
        'TotalTimeMS': 'BIGINT UNSIGNED'
    }, {
        'ScriptName': 'santander_ibis.py',
        'CommitHash': args.commit
    })
'''

# Delete old table
if not args.dnd:
    print("Deleting", database_name ,"old database")
    try:
        conn.drop_database(database_name, force=True)
        time.sleep(2)
        conn = omnisci_server.connect_to_server()
    except Exception as err:
        print("Failed to delete", database_name, "old database: ", err)


print("Creating new database")
try:
	conn.create_database(database_name) # Ibis list_databases method is not supported yet
except Exception as err:
	print("Database creation is skipped, because of error:", err)


if conn.exists_table == False:
    # Create new table
    print("Creating new table", table_name)
    try:
        conn.create_table(table_name = table_name, schema=schema_train, database=database_name)
    except Exception as err:
        print("Failed to create table:", err)

# Create table and import data
if not args.dni:
    # Datafiles import
    t_import_pandas, t_import_ibis = omnisci_server.import_data_by_ibis(data_files_names=args.dp, files_limit=1, columns_names=datafile_columns_names, header=1)
    print("Pandas import time:", t_import_pandas)
    print("Ibis import time:", t_import_ibis)

try:
    db = conn.database(database_name)
except Exception as err:
    print("Failed to connect to database:", err)

try:
    tables_names = db.list_tables()
    print("Database tables:", tables_names)
except Exception as err:
    print("Failed to read database tables:", err)

try:
    df = db.table(table_name)
except Exception as err:
    print("Failed to access", table_name, "table:", err)

omnisci_server.terminate()

'''
# Queries definitions
def q1(df):
    df.groupby('target')[['target']].count().execute()

def q2(df):
    df.groupby('passenger_count').aggregate(total_amount=df.total_amount.mean())[['passenger_count','total_amount']].execute()

def q3(df):
    df.groupby([df.passenger_count, df.pickup_datetime.year().name('pickup_datetime')]).aggregate(count=df.passenger_count.count()).execute()

def q4(df):
    df.groupby([df.passenger_count, df.pickup_datetime.year().name('pickup_datetime'), df.trip_distance]).size().sort_by([('pickup_datetime', True), ('count', False)]).execute()

def timeq(q):
    t = time.time()
    q(df)
    return time.time()-t

queries_description = {1: "query 1"}

def queries_exec(index):
    if index == 1:
        return timeq(q1), queries_description[index]
    elif index == 2:
        return timeq(q2)
    elif index == 3:
        return timeq(q3)
    elif index == 4:
        return timeq(q4)
    else:
        print("Non-valid index value for queries function")
        sys.exit(3)
        return None


try:
    with open(args.r, "w") as report:
        t_begin = time.time()
        for bench_number in range(1,5):
            exec_times = [None]*5
            best_exec_time = float("inf")
            worst_exec_time = 0.0
            first_exec_time = float("inf")
            times_sum = 0.0
            for iteration in range(1, args.i + 1):
                print("RUNNING QUERY NUMBER", bench_number, "ITERATION NUMBER", iteration)
                exec_times[iteration - 1] = int(round(queries_exec(bench_number) * 1000))
                if iteration == 1:
                    first_exec_time = exec_times[iteration - 1]
                if best_exec_time > exec_times[iteration - 1]:
                    best_exec_time = exec_times[iteration - 1]
                if iteration != 1 and worst_exec_time < exec_times[iteration - 1]:
                    worst_exec_time = exec_times[iteration - 1]
                if iteration != 1:
                    times_sum += exec_times[iteration - 1]
            average_exec_time = times_sum/(args.i - 1)
            total_exec_time = int(round((time.time() - t_begin)*1000))
            print("QUERY", bench_number, "EXEC TIME MS", best_exec_time, "TOTAL TIME MS", total_exec_time)
            print("FilesNumber: ", data_files_number,  ",",
                  "QueryName: ",  'Query' + str(bench_number), ",",
                  "FirstExecTimeMS: ", first_exec_time, ",",
                  "WorstExecTimeMS: ", worst_exec_time, ",",
                  "BestExecTimeMS: ", best_exec_time, ",",
                  "AverageExecTimeMS: ", average_exec_time, ",",
                  "TotalTimeMS: ", total_exec_time, ",",
                  "", '\n', file=report, sep='', end='', flush=True)
            if db_reporter is not None:
                db_reporter.submit({
                    'FilesNumber': data_files_number,
                    'QueryName': 'Query' + str(bench_number),
                    'FirstExecTimeMS': first_exec_time,
                    'WorstExecTimeMS': worst_exec_time,
                    'BestExecTimeMS': best_exec_time,
                    'AverageExecTimeMS': average_exec_time,
                    'TotalTimeMS': total_exec_time
                })
except IOError as err:
    print("Failed writing report file", args.r, err)
finally:
    omnisci_server.terminate()
'''
