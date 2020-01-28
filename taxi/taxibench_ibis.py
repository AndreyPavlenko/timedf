from braceexpand import braceexpand
import mysql.connector
import pandas as pd
import subprocess
import argparse
import time
import glob
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from server import OmnisciServer
from report import DbReport
import ibis


class OmnisciServerWorker:

    _imported_pd_df = {}

    def __init__(self, omnisci_server):
        self.omnisci_server = omnisci_server
        self._omnisci_cmd_line = [self.omnisci_server.omnisci_sql_executable] \
            + [str(self.omnisci_server.database_name),
               "-u", self.omnisci_server.user,
               "-p", self.omnisci_server.password] \
                                 + ["--port", str(self.omnisci_server.server_port)]
        self._command_2_import_CSV = "COPY %s FROM '%s' WITH (header='%s');"
        self._conn = None

    def _read_csv_datafile(self, file_name, columns_names, header=None, compression_type='gzip',
                           nrows=200000):
        "Read csv by Pandas. Function returns Pandas DataFrame,\
        which can be used by ibis load_data function"
        
        print("Reading datafile", file_name)
        return pd.read_csv(file_name, compression=compression_type, header=header,
                           names=columns_names, nrows=nrows)
    
    def connect_to_server(self):
        "Connect to Omnisci server using Ibis framework"
        
        self._conn = ibis.omniscidb.connect(host="localhost", port=self.omnisci_server.server_port,
                                            user=self.omnisci_server.user,
                                            password=self.omnisci_server.password)
        return self._conn

    def terminate(self):
        self.omnisci_server.terminate()

    def import_data(self, table_name, data_files_names, files_limit, columns_names, columns_types,
                    header=False):
        "Import CSV files using COPY SQL statement"

        if header:
            header_value = 'true'
        elif not header:
            header_value = 'false'
        else:
            print("Wrong value of header argument!")
            sys.exit(2)
            
        schema_table = ibis.Schema(
            names=columns_names,
            types=columns_types
        )
        
        if not self._conn.exists_table(name=table_name, database=self.omnisci_server.database_name):
            try:
                self._conn.create_table(table_name=table_name, schema=schema_table,
                                        database=self.omnisci_server.database_name)
            except Exception as err:
                print("Failed to create table:", err)

        for f in data_files_names[:files_limit]:
            print("Importing datafile", f)
            copy_str = self._command_2_import_CSV % (table_name, f, header_value)

            try:
                import_process = subprocess.Popen(self._omnisci_cmd_line, stdout=subprocess.PIPE,
                                                  stderr=subprocess.STDOUT, stdin=subprocess.PIPE)
                output = import_process.communicate(copy_str.encode())
            except OSError as err:
                print("Failed to start", self._omnisci_cmd_line, err)

            print(str(output[0].strip().decode()))
            print("Command returned", import_process.returncode)
    
    def import_data_by_ibis(self, table_name, data_files_names, files_limit, columns_names,
                            columns_types, cast_dict, header=None):
        "Import CSV files using Ibis load_data from the Pandas.DataFrame"
        
        schema_table = ibis.Schema(
            names=columns_names,
            types=columns_types
        )
        
        if not self._conn.exists_table(name=table_name, database=self.omnisci_server.database_name):
            try:
                self._conn.create_table(table_name=table_name, schema=schema_table,
                                        database=self.omnisci_server.database_name)
            except Exception as err:
                print("Failed to create table:", err)

        t0 = time.time()
        if files_limit > 1:
            pandas_df_from_each_file = (self._read_csv_datafile(file_name, columns_names, header)
                                        for file_name in data_files_names[:files_limit])
            self._imported_pd_df[table_name] = pd.concat(pandas_df_from_each_file,
                                                         ignore_index=True)
        else:
            self._imported_pd_df[table_name] = self._read_csv_datafile(data_files_names,
                                                                       columns_names, header)
        
        t_import_pandas = time.time() - t0
            
        pandas_concatenated_df_casted = self._imported_pd_df[table_name].astype(dtype=cast_dict,
                                                                                copy=True)

        t0 = time.time()
        self._conn.load_data(table_name=table_name, obj=pandas_concatenated_df_casted,
                             database=self.omnisci_server.database_name)
        t_import_ibis = time.time() - t0

        return t_import_pandas, t_import_ibis
    
    def drop_table(self, table_name):
        "Drop table by table_name using Ibis framework"
        
        if self._conn.exists_table(name=table_name, database=self.omnisci_server.database_name):
            db = self._conn.database(self.omnisci_server.database_name)
            df = db.table(table_name)
            df.drop()
            if table_name in self._imported_pd_df:
                del self._imported_pd_df[table_name]
        else:
            print("Table", table_name, "doesn't exist!")
            sys.exit(3)
    
    def get_pd_df(self, table_name):
        "Get already imported Pandas DataFrame"
        
        if self._conn.exists_table(name=table_name, database=self.omnisci_server.database_name):
            return self._imported_pd_df[table_name]
        else:
            print("Table", table_name, "doesn't exist!")
            sys.exit(4)


omnisci_executable = "../omnisci/build/bin/omnisql"
taxi_trips_directory = "/localdisk/work/trips_x*.csv"
taxibench_table_name = "trips"
omnisci_server = None

parser = argparse.ArgumentParser(description='Run NY Taxi benchmark using Ibis.')

parser.add_argument('-e', default=omnisci_executable, help='Path to executable "omnisci_server".')
parser.add_argument('-r', default="report_taxibench_ibis.csv", help="Report file name.")
parser.add_argument('-df', default=1, type=int,
                    help="Number of datafiles to input into database for processing.")
parser.add_argument('-dp', default=taxi_trips_directory,
                    help="Wildcard pattern of datafiles that should be loaded.")
parser.add_argument('-i', default=5, type=int,
                    help="Number of iterations to run every query. Best result is selected.")
parser.add_argument('-dnd', action='store_true',
                    help="Do not delete old table.")
parser.add_argument('-dni', action='store_true',
                    help="Do not create new table and import any data from CSV files.")
parser.add_argument("-port", default=62074, type=int,
                    help="TCP port that omnisql client should use to connect to server.")
parser.add_argument("-u", default="admin",
                      help="User name to use on omniscidb server.")
parser.add_argument("-p", default="HyperInteractive",
                      help="User password to use on omniscidb server.")
parser.add_argument("-n", default="agent_test_ibis",
                      help="Database name to use on omniscidb server.")
parser.add_argument("-db-server", default="localhost", help="Host name of MySQL server.")
parser.add_argument("-db-port", default=3306, type=int, help="Port number of MySQL server.")
parser.add_argument("-db-user", default="",
                    help="Username to use to connect to MySQL database. If user name is specified,\
                     script attempts to store results in MySQL database using other -db-*\
                      parameters.")
parser.add_argument("-db-pass", default="omniscidb",
                    help="Password to use to connect to MySQL database.")
parser.add_argument("-db-name", default="omniscidb",
                    help="MySQL database to use to store benchmark results.")
parser.add_argument("-db-table", help="Table to use to store results for this benchmark.")

parser.add_argument("-commit_omnisci", dest="commit_omnisci",
                    default="1234567890123456789012345678901234567890",
                    help="Omnisci commit hash to use for tests.")
parser.add_argument("-commit_ibis", dest="commit_ibis",
                    default="1234567890123456789012345678901234567890",
                    help="Ibis commit hash to use for tests.")

args = parser.parse_args()
args.dp = args.dp.replace("'" , "")
if args.df <= 0:
    print("Bad number of data files specified", args.df)
    sys.exit(1)

if args.i < 1:
    print("Bad number of iterations specified", args.i)

database_name = args.n
omnisci_server = OmnisciServer(omnisci_executable=args.e, omnisci_port=args.port,
                               database_name=database_name, user=args.u,
                               password=args.p)
omnisci_server.launch()
omnisci_server_worker = OmnisciServerWorker(omnisci_server)

time.sleep(2)
conn = omnisci_server_worker.connect_to_server()

taxibench_columns_names = ["trip_id", "vendor_id", "pickup_datetime", "dropoff_datetime",
                           "store_and_fwd_flag", "rate_code_id", "pickup_longitude",
                           "pickup_latitude", "dropoff_longitude", "dropoff_latitude",
                           "passenger_count", "trip_distance", "fare_amount", "extra", "mta_tax",
                           "tip_amount", "tolls_amount", "ehail_fee", "improvement_surcharge",
                           "total_amount", "payment_type", "trip_type", "pickup", "dropoff",
                           "cab_type", "precipitation", "snow_depth", "snowfall", "max_temperature",
                           "min_temperature", "average_wind_speed", "pickup_nyct2010_gid",
                           "pickup_ctlabel", "pickup_borocode", "pickup_boroname", "pickup_ct2010",
                           "pickup_boroct2010", "pickup_cdeligibil", "pickup_ntacode",
                           "pickup_ntaname", "pickup_puma", "dropoff_nyct2010_gid",
                           "dropoff_ctlabel", "dropoff_borocode", "dropoff_boroname",
                           "dropoff_ct2010", "dropoff_boroct2010", "dropoff_cdeligibil",
                           "dropoff_ntacode", "dropoff_ntaname", "dropoff_puma"]
taxibench_columns_types = ['int32', 'string', 'timestamp', 'timestamp', 'string', 'int16',
                           'decimal', 'decimal', 'decimal', 'decimal', 'int16', 'decimal',
                           'decimal', 'decimal', 'decimal', 'decimal', 'decimal', 'decimal',
                           'decimal', 'decimal', 'string', 'int16', 'string', 'string', 'string',
                           'int16', 'int16', 'int16', 'int16', 'int16', 'int16', 'int16', 'string',
                           'int16', 'string', 'string', 'string', 'string', 'string', 'string',
                           'string', 'int16', 'string', 'int16', 'string', 'string', 'string',
                           'string', 'string', 'string', 'string']

db_reporter = None
if args.db_user is not "":
    print("Connecting to database")
    db = mysql.connector.connect(host=args.db_server, port=args.db_port, user=args.db_user,
                                 passwd=args.db_pass, db=args.db_name)
    db_reporter = DbReport(db, args.db_table, {
        'FilesNumber': 'INT UNSIGNED NOT NULL',
        'QueryName': 'VARCHAR(500) NOT NULL',
        'FirstExecTimeMS': 'BIGINT UNSIGNED',
        'WorstExecTimeMS': 'BIGINT UNSIGNED',
        'BestExecTimeMS': 'BIGINT UNSIGNED',
        'AverageExecTimeMS': 'BIGINT UNSIGNED',
        'TotalTimeMS': 'BIGINT UNSIGNED'
    }, {
        'ScriptName': 'taxibench_ibis.py',
        'CommitHash': f"{args.commit_omnisci}-{args.commit_ibis}"
    })

# Delete old table
if not args.dnd:
    print("Deleting", database_name ,"old database")
    try:
        conn.drop_database(database_name, force=True)
        time.sleep(2)
        conn = omnisci_server_worker.connect_to_server()
    except Exception as err:
        print("Failed to delete", database_name, "old database: ", err)


data_files_names = list(braceexpand(args.dp))
data_files_names = sorted([x for f in data_files_names for x in glob.glob(f)])
data_files_number = len(data_files_names[:args.df])

try:
    print("Creating", database_name ,"new database")
    conn.create_database(database_name) # Ibis list_databases method is not supported yet
except Exception as err:
    print("Database creation is skipped, because of error:", err)

if len(data_files_names) == 0:
    print("Could not find any data files matching", args.dp)
    sys.exit(2)

# Create table and import data
if not args.dni:
    # Datafiles import
    omnisci_server_worker.import_data(table_name=taxibench_table_name,
                                      data_files_names=data_files_names, files_limit=args.df,
                                      columns_names=taxibench_columns_names,
                                      columns_types=taxibench_columns_types, header=False)

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
    df = db.table(taxibench_table_name)
except Exception as err:
    print("Failed to access", taxibench_table_name,"table:", err)


# Queries definitions
def q1(df):
    df.groupby('cab_type')[['cab_type']].count().execute()


def q2(df):
    df.groupby('passenger_count').aggregate(total_amount=df.total_amount.mean())
    [['passenger_count','total_amount']].execute()


def q3(df):
    df.groupby([df.passenger_count, df.pickup_datetime.year().name('pickup_datetime')])\
        .aggregate(count=df.passenger_count.count()).execute()


def q4(df):
    df.groupby([df.passenger_count, df.pickup_datetime.year().name('pickup_datetime'),
                df.trip_distance]).size().sort_by([('pickup_datetime', True),
                                                   ('count', False)]).execute()


def timeq(q):
    t = time.time()
    q(df)
    return time.time()-t


def queries_exec(index):
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
            print("QUERY", bench_number, "EXEC TIME MS", best_exec_time,
                  "TOTAL TIME MS", total_exec_time)
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
    if omnisci_server:
        omnisci_server.terminate()
