import mysql.connector
import argparse
import time
import sys
import os
import ibis

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from server import OmnisciServer
from report import DbReport
from server_worker import OmnisciServerWorker


def q1():
    t0 = time.time()
    _, _ = omnisci_server_worker.import_data_by_ibis(table_name=tmp_table_name,
                                                     data_files_names=args.dp, files_limit=1,
                                                     columns_names=datafile_columns_names,
                                                     columns_types=datafile_columns_types,
                                                     cast_dict=cast_dict_train, header=0)
    t_import = time.time() - t0
    omnisci_server_worker.drop_table(tmp_table_name)

    return t_import


def q2():
    t_groupby = 0
    for i in range(200):
        col = 'var_%d' % i
        t0 = time.time()
        metric = df[col].count().name('%s_count' % col)
        group_by_expr = df.group_by(col).aggregate(metric)
        _ = group_by_expr.execute()
        t_groupby += time.time() - t0

    return t_groupby


def q3():
    t_where = 0
    global train_where_ibis

    for c, col in enumerate(['var_0', 'var_1', 'var_2']):
        for i in range(1, 4):
            t0 = time.time()
            train_where_ibis[
                train_where_ibis['%s_count' % col] == i].execute()
            t_where += time.time() - t0

    t0 = time.time()
    train_where_ibis[
        train_where_ibis['%s_count' % col] > i].execute()
    t_where += time.time() - t0

    col_to_sel = datafile_columns_names + ["var_" + str(index) + "_count" for index in
                                           range(200)]
    train_where_ibis2 = train_pd_ibis[col_to_sel]
    for i in range(200):
        col = 'var_%d' % i
        t0 = time.time()
        (train_where_ibis2['%s_count' % col] > 1).execute()
        t_where += time.time() - t0

        col_to_sel += ['%s_gt1' % col]
        train_where_ibis2 = train_pd_ibis[col_to_sel]

    return t_where


def q4():
    t0 = time.time()
    # Split operation syntax: OmniSciDBTable[number of rows to split: the last row index of splitted table (last element is not included)]
    train_pd_ibis[190000:190000].execute()
    train_pd_ibis[10000:200000].execute()
    t_split = time.time() - t0

    return t_split


queries_list = [q1, q2, q3, q4]
queries_description = {}
queries_description[1] = 'Santander data file import query'
queries_description[2] = 'Ibis group_gy and count query'
queries_description[3] = 'Rows filtration query'
queries_description[4] = 'Rows split query'

omnisci_executable = "../omnisci/build/bin/omnisci_server"
datafile_directory = "/localdisk/work/train.csv"
train_table_name = "train_table"
omnisci_server = None

parser = argparse.ArgumentParser(description='Run Santander benchmark using Ibis.')

parser.add_argument('-e', default=omnisci_executable, help='Path to executable "omnisql".')
parser.add_argument('-r', default="report_santander_ibis.csv", help="Report file name.")
parser.add_argument('-dp', default=datafile_directory, help="Datafile that should be loaded.")
parser.add_argument('-i', default=5, type=int,
                    help="Number of iterations to run every query. Best result is selected.")
parser.add_argument('-dnd', action='store_true', help="Do not delete old table.")
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
                    help="Username to use to connect to MySQL database. "
                         "If user name is specified, script attempts to store results in "
                         "MySQL database using other -db-* parameters.")
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

try:
    args = parser.parse_args()

    if args.i < 1:
        print("Bad number of iterations specified", args.i)

    datafile_columns_names = ["ID_code", "target"] + ["var_" + str(index) for index in range(200)]
    datafile_columns_types = ["string", "int16"] + ["float32" for _ in range(200)]

    schema_train = ibis.Schema(
        names=datafile_columns_names,
        types=datafile_columns_types
    )

    database_name = args.n
    omnisci_server = OmnisciServer(omnisci_executable=args.e, omnisci_port=args.port,
                                   database_name=database_name, user=args.u,
                                   password=args.p)
    omnisci_server.launch()
    omnisci_server_worker = OmnisciServerWorker(omnisci_server)

    time.sleep(2)
    conn = omnisci_server_worker.connect_to_server()

    db_reporter = None
    if args.db_user is not "":
        print("Connecting to database")
        db = mysql.connector.connect(host=args.db_server, port=args.db_port, user=args.db_user,
                                     passwd=args.db_pass, db=args.db_name)
        db_reporter = DbReport(db, args.db_table, {
            'QueryName': 'VARCHAR(500) NOT NULL',
            'FirstExecTimeMS': 'BIGINT UNSIGNED',
            'WorstExecTimeMS': 'BIGINT UNSIGNED',
            'BestExecTimeMS': 'BIGINT UNSIGNED',
            'AverageExecTimeMS': 'BIGINT UNSIGNED',
            'TotalTimeMS': 'BIGINT UNSIGNED',
            'IbisCommitHash': 'VARCHAR(500) NOT NULL'
        }, {
            'ScriptName': 'santander_ibis.py',
            'CommitHash': args.commit_omnisci
        })

    # Delete old table
    if not args.dnd:
        print("Deleting", database_name, "old database")
        try:
            conn.drop_database(database_name, force=True)
            time.sleep(2)
            conn = omnisci_server_worker.connect_to_server()
        except Exception as err:
            print("Failed to delete", database_name, "old database: ", err)

    print("Creating new database")
    try:
        conn.create_database(database_name)  # Ibis list_databases method is not supported yet
    except Exception as err:
        print("Database creation is skipped, because of error:", err)

    cast_dict_train = {('var_%s' % str(i)): 'float32' for i in range(200)}
    cast_dict_train['target'] = 'int16'

    args.dp = args.dp.replace("'", "")
    if not args.dni:
        # Datafiles import
        t_import_pandas, t_import_ibis = omnisci_server_worker.import_data_by_ibis(
            table_name=train_table_name, data_files_names=args.dp, files_limit=1,
            columns_names=datafile_columns_names, columns_types=datafile_columns_types,
            cast_dict=cast_dict_train, header=0)
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
        df = db.table(train_table_name)
    except Exception as err:
        print("Failed to access", train_table_name, "table:", err)

    # Since OmniSciDB doesn't support JOIN operation for tables with non-integer
    # values, tables for filter and split queries were reproduced by Pandas (as it
    # it was done in the similar Pandas benchmark
    # https://gitlab.devtools.intel.com/jianminl/rapids-response-e2e-workloads/blob/master/e2e/santander/santander_cpu.py)

    train_pd = omnisci_server_worker.get_pd_df(table_name=train_table_name)

    for i in range(200):
        col = 'var_%d' % i
        var_count = train_pd.groupby(col).agg({col: 'count'})
        var_count.columns = ['%s_count' % col]
        var_count = var_count.reset_index()
        train_pd = train_pd.merge(var_count, on=col, how='left')

    for i in range(200):
        col = 'var_%d' % i
        mask = train_pd['%s_count' % col] > 1
        train_pd.loc[mask, '%s_gt1' % col] = train_pd.loc[mask, col]

    datafile_columns_names_train_pd = datafile_columns_names + [
        "var_" + str(index) + "_count" for index in range(200)] + [
        "var_" + str(index) + "_gt1" for index in range(200)]
    datafile_columns_types_train_pd = datafile_columns_types + [
        "float32" for _ in range(200)] + [
        "float32" for _ in range(200)]

    schema_train_pd = ibis.Schema(
        names=datafile_columns_names_train_pd,
        types=datafile_columns_types_train_pd
    )

    cast_dict = {}
    cast_dict['target'] = 'int16'
    for i in range(200):
        cast_dict['var_%s' % str(i)] = 'float32'
        cast_dict['var_%s' % str(i) + '_count'] = 'float32'
        cast_dict['var_%s' % str(i) + '_gt1'] = 'float32'

    train_pd = train_pd.astype(dtype=cast_dict, copy=False)

    conn.create_table(table_name='train_pd_table', schema=schema_train_pd, database=database_name)
    conn.load_data('train_pd_table', train_pd)
    train_pd_ibis = db.table('train_pd_table')

    table_name_where = 'train_where_table'
    datafile_columns_names_train_where = datafile_columns_names + ["var_" + str(index) + "_count"
                                                                   for index in range(200)]
    datafile_columns_types_train_where = datafile_columns_types + ["float32" for _ in range(200)]

    schema_train_where = ibis.Schema(
        names=datafile_columns_names_train_where,
        types=datafile_columns_types_train_where
    )

    train = train_pd.copy()
    train_selected = train[datafile_columns_names_train_where]
    conn.create_table(table_name=table_name_where, schema=schema_train_where,
                      database=database_name)
    conn.load_data(table_name_where, train_selected)
    train_where_ibis = db.table(table_name_where)

    # Queries definitions
    tmp_table_name = 'tmp_table'

    try:
        with open(args.r, "w") as report:
            t_begin = time.time()
            for query_number in range(0, 4):
                exec_times = [None] * 5
                best_exec_time = float("inf")
                worst_exec_time = 0.0
                first_exec_time = float("inf")
                times_sum = 0.0
                for iteration in range(1, args.i + 1):
                    print("Running query number:", query_number + 1, "Iteration number:", iteration)
                    time_tmp = int(round(queries_list[query_number]() * 1000))
                    exec_times[iteration - 1] = time_tmp
                    if iteration == 1:
                        first_exec_time = exec_times[iteration - 1]
                    if best_exec_time > exec_times[iteration - 1]:
                        best_exec_time = exec_times[iteration - 1]
                    if iteration != 1 and worst_exec_time < exec_times[iteration - 1]:
                        worst_exec_time = exec_times[iteration - 1]
                    if iteration != 1:
                        times_sum += exec_times[iteration - 1]
                average_exec_time = times_sum / (args.i - 1)
                total_exec_time = int(round(time.time() - t_begin))
                print("Query", query_number + 1, "Exec time (ms):", best_exec_time,
                      "Total time (s):", total_exec_time)
                print("QueryName: ", queries_description[query_number + 1], ",",
                      "IbisCommitHash", args.commit_ibis, ",",
                      "FirstExecTimeMS: ", first_exec_time, ",",
                      "WorstExecTimeMS: ", worst_exec_time, ",",
                      "BestExecTimeMS: ", best_exec_time, ",",
                      "AverageExecTimeMS: ", average_exec_time, ",",
                      "TotalTimeMS: ", total_exec_time, ",",
                      "", '\n', file=report, sep='', end='', flush=True)
                if db_reporter is not None:
                    db_reporter.submit({
                        'QueryName': queries_description[query_number + 1],
                        'IbisCommitHash': args.commit_ibis,
                        'FirstExecTimeMS': first_exec_time,
                        'WorstExecTimeMS': worst_exec_time,
                        'BestExecTimeMS': best_exec_time,
                        'AverageExecTimeMS': average_exec_time,
                        'TotalTimeMS': total_exec_time
                    })
    except IOError as err:
        print("Failed writing report file", args.r, err)
except Exception as exc:
    print("Failed: ", exc)
finally:
    if omnisci_server:
        omnisci_server.terminate()
