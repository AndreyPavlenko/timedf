import mysql.connector
import pandas as pd
import numpy as np
import subprocess
import argparse
import pathlib
import time
import ibis
import glob
import sys
import os

benchmarks = {
    "MQ01.pd": q1,
    "MQ02.pd": q2,
    "MQ03.pd": q3,
    "MQ04.pd": q4
}

omnisciExecutable  = "build/bin/omnisql"
taxiTripsDirectory = "/localdisk/work/trips_x*.csv"

# Load database reporting functions
pathToReportDir = os.path.join(pathlib.Path(__file__).parent, "..", "report")
print(pathToReportDir)
pathToServerDir = os.path.join(pathlib.Path(__file__).parent, "..", "server")
sys.path.insert(1, pathToReportDir)
sys.path.append(1, pathToServerDir)
import report
import server

def executeProcess(cmdline, cwd=None):
    try:
        process = subprocess.Popen(cmdline, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        out = process.communicate()[0].strip().decode()
        print(out)
    except OSError as err:
        print("Failed to start", cmdline, err)

parser = argparse.ArgumentParser(description='Run NY Taxi benchmark using Ibis')

parser.add_argument('-e', default=omnisciExecutable, help='Path to executable "omnisql"')
parser.add_argument('-r', default="report_pandas.csv", help="Report file name.")
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

time.sleep(2)
conn = ibis.omniscidb.connect(host="localhost", port = args.port, user="admin", password="HyperInteractive")

schema = ibis.Schema(
    names = ["trip_id","vendor_id","pickup_datetime","dropoff_datetime","store_and_fwd_flag","rate_code_id","pickup_longitude","pickup_latitude","dropoff_longitude","dropoff_latitude","passenger_count","trip_distance","fare_amount","extra","mta_tax","tip_amount","tolls_amount","ehail_fee","improvement_surcharge","total_amount","payment_type","trip_type","pickup","dropoff","cab_type","precipitation","snow_depth","snowfall","max_temperature","min_temperature","average_wind_speed","pickup_nyct2010_gid","pickup_ctlabel","pickup_borocode","pickup_boroname","pickup_ct2010","pickup_boroct2010","pickup_cdeligibil","pickup_ntacode","pickup_ntaname","pickup_puma","dropoff_nyct2010_gid","dropoff_ctlabel","dropoff_borocode","dropoff_boroname","dropoff_ct2010","dropoff_boroct2010","dropoff_cdeligibil","dropoff_ntacode","dropoff_ntaname", "dropoff_puma"],
    types = ['int32','string','timestamp','timestamp','string','int16','decimal','decimal','decimal','decimal','int16','decimal','decimal','decimal','decimal','decimal','decimal','decimal','decimal','decimal','string','int16','string','string','string','int16','int16','int16','int16','int16','int16','int16','string','int16','string','string','string','string','string','string','string','int16','string','int16','string','string','string','string','string','string','string']
 )

db_reporter = None
if args.db_user is not "":
    print("Connecting to database")
    db = mysql.connector.connect(host=args.db_server, port=args.db_port, user=args.db_user, passwd=args.db_pass, db=args.db_name);
    db_reporter = report.DbReport(db, args.db_table, {
        'FilesNumber': 'INT UNSIGNED NOT NULL',
        'FragmentSize': 'BIGINT UNSIGNED NOT NULL',
        'BenchName': 'VARCHAR(500) NOT NULL',
        'BestExecTimeMS': 'BIGINT UNSIGNED',
        'BestTotalTimeMS': 'BIGINT UNSIGNED',
        'WorstExecTimeMS': 'BIGINT UNSIGNED',
        'WorstTotalTimeMS': 'BIGINT UNSIGNED',
        'AverageExecTimeMS': 'BIGINT UNSIGNED',
        'AverageTotalTimeMS': 'BIGINT UNSIGNED'
    }, {
        'ScriptName': 'taxibench_pandas.py',
        'CommitHash': args.commit
    })

dataFileNames = sorted(glob.glob(args.dp))
if len(dataFileNames) == 0:
    print("Could not find any data files matching", args.dp)
    sys.exit(2)






print("READING", args.df, "DATAFILES")
dataFilesNumber = len(dataFileNames[:args.df])
def read_datafile(f):
    print("READING DATAFILE", f)
    return pd.read_csv(f, compression='gzip', header=None, names=taxi_names)
df_from_each_file = (read_datafile(f) for f in dataFileNames[:args.df])
concatenated_df = pd.concat(df_from_each_file, ignore_index=True)



try:
    db = conn.database("taxitestdb")
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
def q1():
    df.groupby('cab_type')[['cab_type']].count().execute()

def q2():
    df.groupby('passenger_count').aggregate(total_amount=df.total_amount.mean())[['passenger_count','total_amount']].execute()

def q3():
    df.groupby([df.passenger_count, df.pickup_datetime.year().name('pickup_datetime')]).aggregate(count=df.passenger_count.count()).execute()

def q4():
    df.groupby([df.passenger_count, df.pickup_datetime.year().name('pickup_datetime'), df.trip_distance]).size().sort_by([('pickup_datetime', True), ('count', False)]).execute()

try:
    with open(args.r, "w") as report:
        for benchName, query in benchmarks.items():
            bestExecTime = float("inf")
            for iii in range(1, args.iterations + 1):
                print("RUNNING BENCHMARK NUMBER", benchName, "ITERATION NUMBER", iii)
                query_df = concatenated_df
                t1 = time.time()
                query(query_df)
                t2 = time.time()
                ttt = int(round((t2 - t1) * 1000))
                if bestExecTime > ttt:
                    bestExecTime = ttt
            print("BENCHMARK", benchName, "EXEC TIME", bestExecTime)
            print(dataFilesNumber, ",",
                  0, ",",
                  benchName, ",",
                  bestExecTime, ",",
                  bestExecTime, ",",
                  bestExecTime, ",",
                  bestExecTime, ",",
                  bestExecTime, ",",
                  bestExecTime, ",",
                  "", '\n', file=report, sep='', end='', flush=True)
            if db_reporter is not None:
                db_reporter.submit({
                    'FilesNumber': dataFilesNumber,
                    'FragmentSize': 0,
                    'BenchName': benchName,
                    'BestExecTimeMS': bestExecTime,
                    'BestTotalTimeMS': bestExecTime,
                    'WorstExecTimeMS': bestExecTime,
                    'WorstTotalTimeMS': bestExecTime,
                    'AverageExecTimeMS': bestExecTime,
                    'AverageTotalTimeMS': bestExecTime
                })
except IOError as err:
    print("Failed writing report file", args.r, err)
