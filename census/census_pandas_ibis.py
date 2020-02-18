# coding: utf-8
import os
import sys
import argparse
import warnings
import time
import gzip
import mysql.connector
from timeit import default_timer as timer
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from server import OmnisciServer
from report import DbReport
from server_worker import OmnisciServerWorker

warnings.filterwarnings('ignore')


# Dataset link
# https://rapidsai-data.s3.us-east-2.amazonaws.com/datasets/ipums_education2income_1970-2010.csv.gz
def load_data(
    cached='ipums_education2income_1970-2010.csv.gz', source='ipums'
):
    if os.path.exists(cached) and source == 'ipums':
        with gzip.open(cached) as f:
            X = pd.read_csv(f)
    else:
        print("Dataset: [{}] not found!".format(cached))
        X = None
    return X


def etl_pandas(filename):
    etl_times = {
        't_readcsv': 0.0,
        't_where': 0.0,
        't_arithm': 0.0,
        't_fillna': 0.0,
        't_drop': 0.0,
        't_typeconvert': 0.0,
        't_etl': 0.0,
    }

    t0 = timer()
    df = load_data(filename)
    etl_times['t_readcsv'] = timer() - t0

    t_etl_start = timer()
    df = df.query('INCTOT != 9999999')
    etl_times['t_where'] += timer() - t_etl_start

    t0 = timer()
    df['INCTOT'] = df['INCTOT'] * df['CPI99']
    etl_times['t_arithm'] += timer() - t0

    suspect = [
        'CBSERIAL',
        'EDUC',
        'EDUCD',
        'EDUC_HEAD',
        'EDUC_POP',
        'EDUC_MOM',
        'EDUCD_MOM2',
        'EDUCD_POP2',
        'INCTOT_MOM',
        'INCTOT_POP',
        'INCTOT_MOM2',
        'INCTOT_POP2',
        'INCTOT_HEAD',
    ]
    for column in suspect:
        t0 = timer()
        df[column] = df[column].fillna(-1)
        etl_times['t_fillna'] += timer() - t0

    totincome = ['EDUC', 'EDUCD']
    for column in totincome:
        t0 = timer()
        df = df.query(column + ' != -1')
        etl_times['t_where'] += timer() - t0

    keep_cols = [
        'YEAR0',
        'DATANUM',
        'SERIAL',
        'CBSERIAL',
        'HHWT',
        'GQ',
        'PERNUM',
        'SEX',
        'AGE',
        'INCTOT',
        'EDUC',
        'EDUCD',
        'EDUC_HEAD',
        'EDUC_POP',
        'EDUC_MOM',
        'EDUCD_MOM2',
        'EDUCD_POP2',
        'INCTOT_MOM',
        'INCTOT_POP',
        'INCTOT_MOM2',
        'INCTOT_POP2',
        'INCTOT_HEAD',
        'SEX_HEAD',
    ]
    t0 = timer()
    df = df[keep_cols]
    etl_times['t_drop'] += timer() - t0

    for column in keep_cols:
        t0 = timer()
        df[column] = df[column].fillna(-1)
        etl_times['t_fillna'] += timer() - t0

        t0 = timer()
        df[column] = df[column].astype('float64')
        etl_times['t_typeconvert'] += timer() - t0

    y = df["EDUC"]
    t0 = timer()
    X = df.drop(columns=["EDUC"])
    etl_times['t_drop'] += timer() - t0

    etl_times['t_etl'] = timer() - t_etl_start
    print("DataFrame shape:", df.shape)

    return X, y, etl_times


def etl_ibis(filename, database_name, table_name):
    '''dtypes = OrderedDict([
    ('YEAR',             'int64')
    ('DATANUM',          'int64')
    ('SERIAL',           'int64')
    ('CBSERIAL',       'float64')
    'HHWT'             int64
    CPI99          float64
    GQ               int64
    QGQ              int64
    PERNUM           int64
    PERWT            int64
    SEX              int64
    AGE              int64
    EDUC             int64
    EDUCD            int64
    INCTOT           int64
    SEX_HEAD       float64
    SEX_MOM        float64
    SEX_POP        float64
    SEX_SP         float64
    SEX_MOM2       float64
    SEX_POP2       float64
    AGE_HEAD       float64
    AGE_MOM        float64
    AGE_POP        float64
    AGE_SP         float64
    AGE_MOM2       float64
    AGE_POP2       float64
    EDUC_HEAD      float64
    EDUC_MOM       float64
    EDUC_POP       float64
    EDUC_SP        float64
    EDUC_MOM2      float64
    EDUC_POP2      float64
    EDUCD_HEAD     float64
    EDUCD_MOM      float64
    EDUCD_POP      float64
    EDUCD_SP       float64
    EDUCD_MOM2     float64
    EDUCD_POP2     float64
    INCTOT_HEAD    float64
    INCTOT_MOM     float64
    INCTOT_POP     float64
    INCTOT_SP      float64
    INCTOT_MOM2    float64
    INCTOT_POP2    float64
    ])'''

    etl_times = {
        't_readcsv': 0.0,
        't_where': 0.0,
        't_arithm': 0.0,
        't_fillna': 0.0,
        't_drop': 0.0,
        't_typeconvert': 0.0,
        't_etl': 0.0,
    }

    import ibis

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
        # db_reporter = DbReport(db, args.db_table, {
        #     'FilesNumber': 'INT UNSIGNED NOT NULL',
        #     'QueryName': 'VARCHAR(500) NOT NULL',
        #     'FirstExecTimeMS': 'BIGINT UNSIGNED',
        #     'WorstExecTimeMS': 'BIGINT UNSIGNED',
        #     'BestExecTimeMS': 'BIGINT UNSIGNED',
        #     'AverageExecTimeMS': 'BIGINT UNSIGNED',
        #     'TotalTimeMS': 'BIGINT UNSIGNED',
        #     'QueryValidation': 'VARCHAR(500) NOT NULL',
        #     'IbisCommitHash': 'VARCHAR(500) NOT NULL'
        # }, {
        #     'ScriptName': 'taxibench_ibis.py',
        #                            'CommitHash': args.commit_omnisci
        #                        })

        # Delete old table
    if not args.dnd:
        print("Deleting", database_name, "old database")
        try:
            conn.drop_database(database_name, force=True)
            time.sleep(2)
            conn = omnisci_server_worker.connect_to_server()
        except Exception as err:
            print("Failed to delete", database_name, "old database: ", err)

    try:
        print("Creating", database_name, "new database")
        conn.create_database(database_name)  # Ibis list_databases method is not supported yet
    except Exception as err:
        print("Database creation is skipped, because of error:", err)

    t0 = timer()
    df = load_data(filename)

    # create schema
    fields = [dtype for dtype in df.dtypes.index]
    dtypes = [df.dtypes[field].__str__() for field in fields]
    schema = ibis.Schema(names=fields, types=dtypes)

    # Create table and import data
    if not args.dni:
        # Datafiles import
        omnisci_server_worker.import_data_from_pd_df(table_name=table_name,
                                                     pd_obj=df,
                                                     columns_names=fields,
                                                     columns_types=dtypes)

    etl_times['t_readcsv'] = timer() - t0

    db = conn.database(database_name)
    table = db.table(table_name)

    t_etl_start = timer()
    table = table[table.INCTOT != 9999999]
    etl_times['t_where'] += timer() - t_etl_start

    t0 = timer()
    table = table.set_column('INCTOT', table['INCTOT'] * table['CPI99'])
    etl_times['t_arithm'] += timer() - t0

    suspect = [
        'CBSERIAL',
        'EDUC',
        'EDUCD',
        'EDUC_HEAD',
        'EDUC_POP',
        'EDUC_MOM',
        'EDUCD_MOM2',
        'EDUCD_POP2',
        'INCTOT_MOM',
        'INCTOT_POP',
        'INCTOT_MOM2',
        'INCTOT_POP2',
        'INCTOT_HEAD',
    ]
    for column in suspect:
        t0 = timer()
        table = table.set_column(column, table[column].fillna(-1))
        etl_times['t_fillna'] += timer() - t0

    t0 = timer()
    table = table[table.EDUC != -1]
    table = table[table.EDUCD != -1]
    etl_times['t_where'] += timer() - t0

    keep_cols = [
        'YEAR0',
        'DATANUM',
        'SERIAL',
        'CBSERIAL',
        'HHWT',
        'GQ',
        'PERNUM',
        'SEX',
        'AGE',
        'INCTOT',
        'EDUC',
        'EDUCD',
        'EDUC_HEAD',
        'EDUC_POP',
        'EDUC_MOM',
        'EDUCD_MOM2',
        'EDUCD_POP2',
        'INCTOT_MOM',
        'INCTOT_POP',
        'INCTOT_MOM2',
        'INCTOT_POP2',
        'INCTOT_HEAD',
        'SEX_HEAD',
    ]
    t0 = timer()
    table = table[keep_cols]
    etl_times['t_drop'] += timer() - t0

    for column in keep_cols:
        t0 = timer()
        table = table.set_column(column, table[column].fillna(-1))
        etl_times['t_fillna'] += timer() - t0

        t0 = timer()
        table = table.set_column(column, table[column].cast('float64'))
        etl_times['t_typeconvert'] += timer() - t0

    y = table["EDUC"].execute()
    t0 = timer()
    X = table.drop(["EDUC"]).execute()
    etl_times['t_drop'] += timer() - t0

    etl_times['t_etl'] = timer() - t_etl_start
    print("DataFrame shape:", df.shape)

    # ibis_df = table.execute()
    # pd.testing.assert_frame_equal(ibis_df, df)

    return X, y, etl_times


def print_times(etl_times, name=None):
    if name:
        print(f"{name} times:")
    for time_name, time in etl_times.items():
        print('{} = {:.5f} s'.format(time_name, time))


def mse(y_test, y_pred):
    return ((y_test - y_pred) ** 2).mean()


def cod(y_test, y_pred):
    y_bar = y_test.mean()
    total = ((y_test - y_bar) ** 2).sum()
    residuals = ((y_test - y_pred) ** 2).sum()
    return 1 - (residuals / total)


def ml(X, y, random_state, n_runs, train_size):
    if len(sys.argv) == 3:
        print("Intel optimized sklearn is used")
        from daal4py.sklearn.model_selection import train_test_split
        import daal4py.sklearn.linear_model as lm
    else:
        print("Stock sklearn is used")
        from sklearn.model_selection import train_test_split
        import sklearn.linear_model as lm

    clf = lm.Ridge()

    mse_values, cod_values = [], []
    ml_times = {'t_ML': 0.0, 't_train': 0.0, 't_inference': 0.0}

    print('ML runs: ', n_runs)
    for i in range(n_runs):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=train_size, random_state=random_state
        )
        random_state += 777

        t0 = timer()
        model = clf.fit(X_train, y_train)
        ml_times['t_train'] += timer() - t0

        t0 = timer()
        y_pred = model.predict(X_test)
        ml_times['t_inference'] += timer() - t0

        mse_values.append(mse(y_test, y_pred))
        cod_values.append(cod(y_test, y_pred))

    ml_times['t_ML'] += ml_times['t_train'] + ml_times['t_inference']

    mse_mean = sum(mse_values) / len(mse_values)
    cod_mean = sum(cod_values) / len(cod_values)
    mse_dev = pow(
        sum([(mse_value - mse_mean) ** 2 for mse_value in mse_values])
        / (len(mse_values) - 1),
        0.5,
    )
    cod_dev = pow(
        sum([(cod_value - cod_mean) ** 2 for cod_value in cod_values])
        / (len(cod_values) - 1),
        0.5,
    )

    return mse_mean, cod_mean, mse_dev, cod_dev, ml_times


def main():
    omniscript_path = os.path.dirname(__file__)
    omnisci_server = None
    args = None

    parser = argparse.ArgumentParser(description='Run internal tests from ibis project')
    optional = parser._action_groups.pop()
    required = parser.add_argument_group("required arguments")
    parser._action_groups.append(optional)

    required.add_argument("-f", "--file", dest="file", required=True,
                          help="A datafile that should be loaded")
    optional.add_argument('-dnd', action='store_true',
                          help="Do not delete old table.")
    optional.add_argument('-dni', action='store_true',
                          help="Do not create new table and import any data from CSV files.")
    optional.add_argument('-val', action='store_true',
                          help="validate queries results (by comparison with Pandas queries results).")
    optional.add_argument('-o', '--optimizer', dest='optimizer', default='intel',
                          help="Which optimizer is used")
    # MySQL database parameters
    optional.add_argument('-db-server', dest="db_server", default="localhost",
                          help="Host name of MySQL server.")
    optional.add_argument('-db-port', dest="db_port", default=3306, type=int,
                          help="Port number of MySQL server.")
    optional.add_argument('-db-user', dest="db_user", default="",
                          help="Username to use to connect to MySQL database. "
                               "If user name is specified, script attempts to store results in MySQL "
                               "database using other -db-* parameters.")
    optional.add_argument('-db-pass', dest="db_password", default="omniscidb",
                          help="Password to use to connect to MySQL database.")
    optional.add_argument('-db-name', dest="db_name", default="omniscidb",
                          help="MySQL database to use to store benchmark results.")
    optional.add_argument('-db-table', dest="db_table",
                          help="Table to use to store results for this benchmark.")
    # Omnisci server parameters
    optional.add_argument("-e", "--executable", dest="omnisci_executable", required=True,
                          help="Path to omnisci_server executable.")
    optional.add_argument("-w", "--workdir", dest="omnisci_cwd",
                          help="Path to omnisci working directory. "
                               "By default parent directory of executable location is used. "
                               "Data directory is used in this location.")
    optional.add_argument("-port", "--omnisci_port", dest="omnisci_port", default=6274, type=int,
                          help="TCP port number to run omnisci_server on.")
    optional.add_argument("-u", "--user", dest="user", default="admin",
                          help="User name to use on omniscidb server.")
    optional.add_argument("-p", "--password", dest="password", default="HyperInteractive",
                          help="User password to use on omniscidb server.")
    optional.add_argument("-n", "--name", dest="name", default="census_database",
                          help="Database name to use in omniscidb server.")
    optional.add_argument("-t", "--table", dest="table", default="census_table",
                          help="Table name name to use in omniscidb server.")

    optional.add_argument("-commit_omnisci", dest="commit_omnisci",
                          default="1234567890123456789012345678901234567890",
                          help="Omnisci commit hash to use for tests.")
    optional.add_argument("-commit_ibis", dest="commit_ibis",
                          default="1234567890123456789012345678901234567890",
                          help="Ibis commit hash to use for tests.")

    args = parser.parse_args()

    global CENSUS_DATABASE, CENSUS_TABLE
    global N_RUNS, RANDOM_STATE, TRAIN_SIZE

    CENSUS_DATABASE = args.name
    CENSUS_TABLE = args.table

    # ML specific
    N_RUNS = 50
    TRAIN_SIZE = 0.9
    RANDOM_STATE = 777

    if args.optimizer == 'intel':
        print("Intel optimized sklearn is used")
        from daal4py.sklearn.model_selection import train_test_split
        import daal4py.sklearn.linear_model as lm
        clf = lm.Ridge()
    if args.optimizer == 'stock':
        print("Stock sklearn is used")
        from sklearn.model_selection import train_test_split
        import sklearn.linear_model as lm
        clf = lm.Ridge()
    else:
        print(
            f"Intel optimized and stock sklearn are supported. {args.optimizer} can't be recognized'")
        sys.exit(1)

    try:
        X, y, etl_times = etl_ibis(args.file, CENSUS_DATABASE, CENSUS_TABLE)
        print_times(etl_times)
        mse_mean, cod_mean, mse_dev, cod_dev, ml_times = ml(
            X, y, RANDOM_STATE, N_RUNS, TRAIN_SIZE
        )
        print_times(ml_times)
        print("mean MSE ± deviation: {:.9f} ± {:.9f}".format(mse_mean, mse_dev))
        print("mean COD ± deviation: {:.9f} ± {:.9f}".format(cod_mean, cod_dev))

        X, y, etl_times = etl_pandas(args.file)
        print_times(etl_times)
        mse_mean, cod_mean, mse_dev, cod_dev, ml_times = ml(
            X, y, RANDOM_STATE, N_RUNS, TRAIN_SIZE
        )
        print_times(ml_times)
        print("mean MSE ± deviation: {:.9f} ± {:.9f}".format(mse_mean, mse_dev))
        print("mean COD ± deviation: {:.9f} ± {:.9f}".format(cod_mean, cod_dev))\

    except Exception as err:
        print("Failed", err)
        sys.exit(1)
    finally:
        if omnisci_server:
            omnisci_server.terminate()


if __name__ == '__main__':
    main()
