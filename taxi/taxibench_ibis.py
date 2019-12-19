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

omnisciExecutable  = "build/bin/omnisql"
taxiTripsDirectory = "/localdisk/work/trips_x*.csv"
databaseName = "taxibenchdb"

# Load database reporting, server and Ibis packages
pathToReportDir = os.path.join(pathlib.Path(__file__).parent, "..", "report")
print(pathToReportDir)
pathToServerDir = os.path.join(pathlib.Path(__file__).parent, "..", "server")
pathToIbisDir = os.path.join(pathlib.Path(__file__).parent.parent, "..", "ibis")
sys.path.insert(1, pathToReportDir)
sys.path.insert(1, pathToServerDir)
sys.path.insert(1, pathToIbisDir)
import report
import server
import ibis

def executeProcess(cmdline, cwd=None):
    try:
        process = subprocess.Popen(cmdline, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        out = process.communicate()[0].strip().decode()
        print(out)
    except OSError as err:
        print("Failed to start", cmdline, err)

parser = argparse.ArgumentParser(description='Run NY Taxi benchmark using Ibis')

parser.add_argument('-e', default=omnisciExecutable, help='Path to executable "omnisql"')
parser.add_argument('-r', default="report_ibis.csv", help="Report file name.")
parser.add_argument('-df', default=1, type=int, help="Number of datafiles to input into database for processing.")
parser.add_argument('-dp', default=taxiTripsDirectory, help="Wildcard pattern of datafiles that should be loaded.")
parser.add_argument('-i', default=5, type=int, help="Number of iterations to run every benchmark. Best result is selected.")
parser.add_argument('-dnd', action='store_true', help="Do not delete old table")
parser.add_argument('-dni', action='store_true', help="Do not create new table and import any data from CSV files")
parser.add_argument("-port", default=62074, type=int, help="TCP port that omnisql client should use to connect to server")

parser.add_argument("-db-server", default="localhost", help="Host name of MySQL server")
parser.add_argument("-db-port", default=3306, type=int, help="Port number of MySQL server")
parser.add_argument("-db-user", default="", help="Username to use to connect to MySQL database. If user name is specified, script attempts to store results in MySQL database using other -db-* parameters.")
parser.add_argument("-db-pass", default="omniscidb", help="Password to use to connect to MySQL database")
parser.add_argument("-db-name", default="omniscidb", help="MySQL database to use to store benchmark results")
parser.add_argument("-db-table", help="Table to use to store results for this benchmark.")

parser.add_argument("-commit", default="1234567890123456789012345678901234567890", help="Commit hash to use to record this benchmark results")

args = parser.parse_args()

if args.df <= 0:
    print("Bad number of data files specified", args.df)
    sys.exit(1)

if args.i < 1:
    print("Bad number of iterations specified", args.t)

omnisciServer = server.OmnisciServer(omnisci_executable=args.e, omnisci_port=args.port)
omnisciServer.launch()

time.sleep(2)
conn = omnisciServer.connect_to_server()

schema = ibis.Schema(
    names = ["trip_id","vendor_id","pickup_datetime","dropoff_datetime","store_and_fwd_flag","rate_code_id","pickup_longitude","pickup_latitude","dropoff_longitude","dropoff_latitude","passenger_count","trip_distance","fare_amount","extra","mta_tax","tip_amount","tolls_amount","ehail_fee","improvement_surcharge","total_amount","payment_type","trip_type","pickup","dropoff","cab_type","precipitation","snow_depth","snowfall","max_temperature","min_temperature","average_wind_speed","pickup_nyct2010_gid","pickup_ctlabel","pickup_borocode","pickup_boroname","pickup_ct2010","pickup_boroct2010","pickup_cdeligibil","pickup_ntacode","pickup_ntaname","pickup_puma","dropoff_nyct2010_gid","dropoff_ctlabel","dropoff_borocode","dropoff_boroname","dropoff_ct2010","dropoff_boroct2010","dropoff_cdeligibil","dropoff_ntacode","dropoff_ntaname", "dropoff_puma"],
    types = ['int32','string','timestamp','timestamp','string','int16','decimal','decimal','decimal','decimal','int16','decimal','decimal','decimal','decimal','decimal','decimal','decimal','decimal','decimal','string','int16','string','string','string','int16','int16','int16','int16','int16','int16','int16','string','int16','string','string','string','string','string','string','string','int16','string','int16','string','string','string','string','string','string','string']
 )

db_reporter = None
if args.db_user is not "":
    print("Connecting to database")
    db = mysql.connector.connect(host=args.db_server, port=args.db_port, user=args.db_user, passwd=args.db_pass, db=args.db_name)
    db_reporter = report.DbReport(db, args.db_table, {
        'FilesNumber': 'INT UNSIGNED NOT NULL',
        'BenchName': 'VARCHAR(500) NOT NULL',
        'FirstExecTimeMS': 'BIGINT UNSIGNED',
        'WorstExecTimeMS': 'BIGINT UNSIGNED',
        'BestExecTimeMS': 'BIGINT UNSIGNED',
        'AverageExecTimeMS': 'BIGINT UNSIGNED',
        'TotalTimeMS': 'BIGINT UNSIGNED'
    }, {
        'ScriptName': 'taxibench_ibis.py',
        'CommitHash': args.commit
    })

# Delete old table
if not args.dnd:
    print("Deleting", databaseName ,"old database")
    try:
        conn.drop_database(databaseName, force=True)
    except Exception as err:
        print("Failed to delete", databaseName, "old database: ", err)

dataFilesNumber = 0
# Create table and import data
if not args.dni:
    print("Creating new database")
    try:
        conn.create_database(databaseName) # Ibis list_databases method is not supported yet
    except Exception as err:
        print("Database creation is skipped, because of error:", err)

    dataFileNames = sorted(glob.glob(args.dp))
    if len(dataFileNames) == 0:
        print("Could not find any data files matching", args.dp)
        sys.exit(2)

    # Create new table
    print("Creating new table trips")
    try:
        conn.create_table(table_name = "trips", schema=schema, database=databaseName)
    except Exception as err:
        print("Failed to create table: ", err)

    # Datafiles import
    dataFilesNumber = len(dataFileNames[:args.df])
    omnisciServer.import_data(dataFileNames, args.df)

try:
    db = conn.database(databaseName)
except Exception as err:
    print("Failed to connect to database: ", err)

try:
    tablesNames = db.list_tables()
    print(tablesNames)
except Exception as err:
    print("Failed to read database tables: ", err)

if len(tablesNames) != 1 or tablesNames[0] != 'trips':
    print("Database table created with mistake")
    sys.exit(3)

df = db.table('trips')
def q1(df):
    df.groupby('cab_type')[['cab_type']].count().execute()

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

def queriesExec(index):
    if index == 1:
        return timeq(q1)
    elif index == 2:
        return timeq(q2)
    elif index == 3:
        return timeq(q3)
    elif index == 4:
        return timeq(q4)
    else:
        print("Non-valid index value for queries function")
        sys.exit(4)
        return None

try:
    exec_times = []
    with open(args.r, "w") as report:
        t_begin = time.time()
        for benchNumber in range(1,5):
            bestExecTime = float("inf")
            worstExecTime = 0.0
            firstExecTime = float("inf")
            times_sum = 0.0
            for iteration in range(1, args.i + 1):
                print("RUNNING BENCHMARK NUMBER", benchNumber, "ITERATION NUMBER", iteration)
                exec_times[iteration - 1] = int(round(queriesExec(benchNumber) * 1000))
                if iteration == 1:
                    firstExecTime = exec_times[iteration - 1]
                if bestExecTime > exec_times[iteration - 1]:
                    bestExecTime = exec_times[iteration - 1]
                if iteration != 1 and worstExecTime < exec_times[iteration - 1]:
                    worstExecTime = exec_times[iteration - 1]
                if iteration != 1:
                    times_sum += exec_times[iteration - 1]
            averageExecTime = times_sum/(args.i - 1)
            totalExecTime = time.time() - t_begin
            print("BENCHMARK", benchNumber, "EXEC TIME", bestExecTime, "TOTAL TIME", totalExecTime)
            print("FilesNumber: ", dataFilesNumber,  ",",
                  "BenchName: ",  'Benchmark' + benchNumber, ",",
                  "FirstExecTimeMS: ", firstExecTime, ",",
                  "WorstExecTimeMS: ", worstExecTime, ",",
                  "BestExecTimeMS: ", bestExecTime, ",",
                  "AverageExecTimeMS: ", averageExecTime, ",",
                  "TotalTimeMS: ", totalExecTime, ",",
                  "", '\n', file=report, sep='', end='', flush=True)
            if db_reporter is not None:
                db_reporter.submit({
                    'FilesNumber': dataFilesNumber,
                    'BenchName': 'Benchmark' + benchNumber,
                    'FirstExecTimeMS': firstExecTime,
                    'WorstExecTimeMS': worstExecTime,
                    'BestExecTimeMS': bestExecTime,
                    'AverageExecTimeMS': averageExecTime,
                    'TotalTimeMS': totalExecTime
                })
except IOError as err:
    print("Failed writing report file", args.r, err)
finally:
    omnisciServer.terminate()
