#!/usr/bin/env python3
import sys
import time
from collections import OrderedDict

import pandas as pd
import numpy as np
import xgboost as xgb

import ibis
import pymapd


def _parse_dtyped_csv(fname, dtypes, **kw):
    all_but_dates = {col: valtype for (col, valtype) in dtypes.items()
                     if valtype != 'datetime64'}
    dates_only = [col for (col, valtype) in dtypes.items()
                     if valtype == 'datetime64']

    return pd.read_csv(fname, dtype=all_but_dates, parse_dates=dates_only, **kw)

def cpu_load_acquisition_csv(acquisition_path, **kwargs):
    """ Loads acquisition data

    Returns
    -------
    GPU DataFrame
    """
    
    cols = [
        'loan_id', 'orig_channel', 'seller_name', 'orig_interest_rate', 'orig_upb', 'orig_loan_term', 
        'orig_date', 'first_pay_date', 'orig_ltv', 'orig_cltv', 'num_borrowers', 'dti', 'borrower_credit_score', 
        'first_home_buyer', 'loan_purpose', 'property_type', 'num_units', 'occupancy_status', 'property_state',
        'zip', 'mortgage_insurance_percent', 'product_type', 'coborrow_credit_score', 'mortgage_insurance_type', 
        'relocation_mortgage_indicator', 'year_quarter_ignore'
    ]
    
    dtypes = OrderedDict([
        ("loan_id", "int64"),
        ("orig_channel", "category"),
        ("seller_name", "category"),
        ("orig_interest_rate", "float64"),
        ("orig_upb", "int64"),
        ("orig_loan_term", "int64"),
        ("orig_date", "datetime64"),
        ("first_pay_date", "datetime64"),
        ("orig_ltv", "float64"),
        ("orig_cltv", "float64"),
        ("num_borrowers", "float64"),
        ("dti", "float64"),
        ("borrower_credit_score", "float64"),
        ("first_home_buyer", "category"),
        ("loan_purpose", "category"),
        ("property_type", "category"),
        ("num_units", "int64"),
        ("occupancy_status", "category"),
        ("property_state", "category"),
        ("zip", "int64"),
        ("mortgage_insurance_percent", "float64"),
        ("product_type", "category"),
        ("coborrow_credit_score", "float64"),
        ("mortgage_insurance_type", "float64"),
        ("relocation_mortgage_indicator", "category"),
        ('year_quarter_ignore', 'int32')
    ]) 
    print(acquisition_path)
    return _parse_dtyped_csv(acquisition_path, dtypes, names=cols, delimiter='|', index_col=False)

def cpu_load_performance_csv(performance_path, **kwargs):
    """ Loads performance data

    Returns
    -------
    GPU DataFrame
    """
    
    cols = [
        "loan_id", "monthly_reporting_period", "servicer", "interest_rate", "current_actual_upb",
        "loan_age", "remaining_months_to_legal_maturity", "adj_remaining_months_to_maturity",
        "maturity_date", "msa", "current_loan_delinquency_status", "mod_flag", "zero_balance_code",
        "zero_balance_effective_date", "last_paid_installment_date", "foreclosed_after",
        "disposition_date", "foreclosure_costs", "prop_preservation_and_repair_costs",
        "asset_recovery_costs", "misc_holding_expenses", "holding_taxes", "net_sale_proceeds",
        "credit_enhancement_proceeds", "repurchase_make_whole_proceeds", "other_foreclosure_proceeds",
        "non_interest_bearing_upb", "principal_forgiveness_upb", "repurchase_make_whole_proceeds_flag",
        "foreclosure_principal_write_off_amount", "servicing_activity_indicator"
    ]
    
    dtypes = OrderedDict([
        ("loan_id", "int64"),
        ("monthly_reporting_period", "datetime64"),
        ("servicer", "category"),
        ("interest_rate", "float64"),
        ("current_actual_upb", "float64"),
        ("loan_age", "float64"),
        ("remaining_months_to_legal_maturity", "float64"),
        ("adj_remaining_months_to_maturity", "float64"),
        ("maturity_date", "datetime64"),
        ("msa", "float64"),
        ("current_loan_delinquency_status", "int32"),
        ("mod_flag", "category"),
        ("zero_balance_code", "category"),
        ("zero_balance_effective_date", "datetime64"),
        ("last_paid_installment_date", "datetime64"),
        ("foreclosed_after", "datetime64"),
        ("disposition_date", "datetime64"),
        ("foreclosure_costs", "float64"),
        ("prop_preservation_and_repair_costs", "float64"),
        ("asset_recovery_costs", "float64"),
        ("misc_holding_expenses", "float64"),
        ("holding_taxes", "float64"),
        ("net_sale_proceeds", "float64"),
        ("credit_enhancement_proceeds", "float64"),
        ("repurchase_make_whole_proceeds", "float64"),
        ("other_foreclosure_proceeds", "float64"),
        ("non_interest_bearing_upb", "float64"),
        ("principal_forgiveness_upb", "float64"),
        ("repurchase_make_whole_proceeds_flag", "category"),
        ("foreclosure_principal_write_off_amount", "float64"),
        ("servicing_activity_indicator", "category")
    ])

    print(performance_path)
    
    return _parse_dtyped_csv(performance_path, dtypes, names=cols, delimiter='|')

#------------------------------------------------------------------------------------------
DB_NAME = 'mortgage'

#def make_conn(host='localhost', port=6274, user='admin', password='HyperInteractive', ipc=None):
def make_conn(host='localhost', port=8000, user='admin', password='HyperInteractive', ipc=None):
    _opts = dict(host=host, port=port, user=user, password=password, ipc=ipc)
    conn = ibis.omniscidb.connect(**_opts)
    conn._opts = _opts
    return conn

def make_db(conn, dbname=DB_NAME, recreate=True):
    try:
        db = conn.database(dbname)
    except pymapd.exceptions.Error as e:
        if 'does not exist' not in e.args[0]:
            raise
        # db doesn't exist, ignore
        db = None
    else:
        # db exists, flush it to start from scratch
        if recreate:
            print('recreating %s db' % dbname)
            conn.drop_database(dbname, force=True)
            time.sleep(2)
            conn = make_conn(**conn._opts)
            db = None

    if not db:
        conn.create_database(dbname)
        db = conn.database(dbname)
    db._name = dbname

    return db, conn

def make_table(conn, database_name, table_name, schema_table):
    if not conn.exists_table(name=table_name, database=database_name):
        conn.create_table(table_name=table_name, schema=schema_table, database=database_name)
    db, conn = make_db(conn, database_name, False)
    table = db.table(table_name)
    table._name = table_name
    return table, conn

#------------------------------------------------------------------------------------------
'''
def load_table(db, table_name, fpath, options=None):
    cmd = "COPY %s FROM '%s'" % (table_name, fpath)
    if options:
        cmd = '%s WITH (%s)' % (cmd, ', '.join('%s=%s' % (k, v) for (k, v) in options.items()))
    db._execute(cmd)
'''

def pd_load_table(conn, db, table_name, pd_df):
    conn.load_data(table_name=table_name, obj=pd_df, database=db._name)
    return db.table(table_name)

class Timer:
    def __init__(self, name):
        self.name = name
    def __enter__(self):
        self.start = time.time()
        return self
    def __exit__(self, *a, **kw):
        print('%s took %.3f sec' % (self.name, time.time() - self.start))

#------------------------------------------------------------------------------------------
def create_joined_df(perf_table):
    delinquency_12_expr = ibis.case().when(perf_table['current_loan_delinquency_status'].notnull(),
                                           perf_table['current_loan_delinquency_status']).else_(-1).end()
    upb_12_expr = ibis.case().when(perf_table['current_actual_upb'].notnull(),
                                   perf_table['current_actual_upb']).else_(999999999).end()
    joined_df = perf_table['loan_id',
                        perf_table['monthly_reporting_period'].month().name('timestamp_month').cast('int32'),
                        perf_table['monthly_reporting_period'].year().name('timestamp_year').cast('int32'),
                        delinquency_12_expr.name('delinquency_12'),
                        upb_12_expr.name('upb_12')]
    return joined_df

def create_12_mon_features(joined_df):
    delinq_df = None
    n_months = 1 # should be 12 but we don't have UNION yet :(
    for y in range(1, n_months + 1):
        year_dec = ibis.case().when(joined_df['timestamp_month'] < ibis.literal(y), 1).else_(0).end()
        tmp_df = joined_df['loan_id', 'delinquency_12', 'upb_12',
                           (joined_df['timestamp_year'] - year_dec).name('timestamp_year')]

        delinquency_12 = (tmp_df['delinquency_12'].max() > 3).cast('int32') + (tmp_df['upb_12'].min() == 0).cast('int32')
        tmp_df = tmp_df.groupby(['loan_id', 'timestamp_year']).aggregate(delinquency_12.name('delinquency_12'))

        tmp_df = tmp_df.mutate(timestamp_month=ibis.literal(y, 'int32'))

        if delinq_df is None:
            delinq_df = tmp_df
        else:
            delinq_df = delinq_df.union(tmp_df)

    return delinq_df

def final_performance_delinquency(perf_table, mon12_df):
    # rename columns, or join fails because it has overlapping keys
    return perf_table.left_join(mon12_df.relabel({'loan_id': 'mon12_loan_id'}), [('loan_id', 'mon12_loan_id'),
                                            perf_table['monthly_reporting_period'].month().cast('int32') == mon12_df['timestamp_month'],
                                            perf_table['monthly_reporting_period'].year().cast('int32') == mon12_df['timestamp_year']])[perf_table, mon12_df['delinquency_12']]


def join_perf_acq_gdfs(perf_df, acq_table):
    merged = perf_df.inner_join(acq_table, ['loan_id'])

    dropList = {
        'loan_id', 'orig_date', 'first_pay_date', 'seller_name',
        'monthly_reporting_period', 'last_paid_installment_date', 'maturity_date', 'ever_30', 'ever_90', 'ever_180',
        'delinquency_30', 'delinquency_90', 'delinquency_180', 'upb_12',
        'zero_balance_effective_date','foreclosed_after', 'disposition_date','timestamp'
    }

    resultCols = []
    for req in (perf_df, acq_table):
        schema = req.schema()
        for colName in schema:
            if colName in dropList:
                continue
            if isinstance(schema[colName], ibis.expr.datatypes.Category):
                resultCols.append(req[colName].cast('int32'))
            else:
                resultCols.append(req[colName])
    return merged[resultCols]

def import_data(conn, db, acq_table, perf_table,
                      quarter=1, year=2000, perf_file="", **kwargs):
    shmem_conn_opts = dict(conn._opts)
    shmem_conn_opts['ipc'] = True
    shmem_conn = make_conn(**shmem_conn_opts)
    shmem_db, shmem_conn = make_db(shmem_conn, recreate=False)
    #load_table(db, names_table._name, col_names_path)

    # print('names.memory_usage')
    # print(names.memory_usage(index=False))

    with Timer('load acqiusition via pandas->ibis'):
        pd_acq_df = cpu_load_acquisition_csv(acquisition_path='%s/Acquisition_%sQ%s.txt' % (acq_data_path, year, quarter))
        with Timer('\tpandas->ibis acq import'):
            acq_table = pd_load_table(shmem_conn, shmem_db, acq_table._name, pd_acq_df)
        del pd_acq_df

    #acq_df = acq_table.left_join(names_table, )
    # print('acq_gdf.memory_usage')
    # print(acq_gdf.memory_usage(index=False))

    #acq_gdf = acq_gdf.merge(names, how='left', on=['seller_name'])
    #acq_gdf = acq_gdf.drop(['seller_name'], axis=1)
    #acq_gdf['seller_name'] = acq_gdf['new']
    #acq_gdf = acq_gdf.drop(['new'], axis=1)

    with Timer('load performance via pandas->ibis'):
        pd_perf_df = cpu_load_performance_csv(perf_file)
        with Timer('\tpandas->ibis perf import'):
            perf_table = pd_load_table(shmem_conn, shmem_db, perf_table._name, pd_perf_df)
        del pd_perf_df
    
    return acq_table, perf_table

# vnlitvin: PoC

def Query_execute(self, **kwargs):
    # inlining omnisci cursor._fetch and hacking it; also hacking execute
    with self.client._execute(
        self.compiled_sql, results=True, **kwargs
    ) as cur:
        import pdb;pdb.set_trace()
        result = cur.to_df()
        result = self.schema().apply_to(result)

    return result

import pymapd._parsers as pymapd_parsers
def pymapd_load_data(buf, schema, tdf=None):
    """
    Load a `pandas.DataFrame` from a buffer written to shared memory

    Parameters
    ----------
    buf : pyarrow.Buffer
    shcema : pyarrow.Schema
    tdf(optional) : TDataFrame

    Returns
    -------
    df : pandas.DataFrame
    """
    print("vnlitvin HACKED it")
    message = pymapd_parsers.pa.read_message(buf)
    rb = pymapd_parsers.pa.read_record_batch(message, schema)
    df = rb #.to_pandas()
    df.set_tdf = pymapd_parsers.MethodType(pymapd_parsers.set_tdf, df)
    df.get_tdf = pymapd_parsers.MethodType(pymapd_parsers.get_tdf, df)
    df.set_tdf(tdf)
    return df

# patching
#ibis.client.Query.execute = Query_execute
#pymapd_parsers._load_data = pymapd_load_data
#import pymapd.connection
#pymapd.connection._load_data = pymapd_load_data
# /PoC

def run_ibis_workflow(acq_table, perf_table):
    with Timer('make ibis queries'):
        joined_df = create_joined_df(perf_table)

        mon12_df = create_12_mon_features(joined_df)
        #del(testdf)

        perf_df = final_performance_delinquency(perf_table, mon12_df)
        #del(gdf, joined_df)

        final_gdf = join_perf_acq_gdfs(perf_df, acq_table)
        #del(perf_df)
        #del(acq_gdf)

        #final_gdf = last_mile_cleaning(final_gdf)

    with Timer('ibis compilation'):
        tmp = final_gdf.compile()

    with Timer('execute queries'):
        result = final_gdf.execute()
    return result


#------------------------------------------------------------------------------------------
t_train = 0
t_dmatrix = 0

def train_xgb(pd_df):
    global t_train, t_dmatrix
    print('xgboost')

    dxgb_cpu_params = {
        'nround':            100,
        'max_depth':         8,
        'max_leaves':        2**8,
        'alpha':             0.9,
        'max_bin' : 256,
        #'eta':               0.1,
        #'gamma':             0.1,
        'learning_rate':     0.1,
        'subsample':         1,
        'reg_lambda':        1,
        'scale_pos_weight':  2,
        'min_child_weight':  0,  #30
        'tree_method':       'hist',
        'predictor' : 'cpu_predictor',
        #n_gpus':            1,
        # 'distributed_dask':  True,
        #'loss':              'ls',
        #'objective':         'reg:linear',
        #'max_features':      'auto',
        #'criterion':         'friedman_mse',
        #'grow_policy':       'lossguide',
        #'verbose':           True
    }

    t0 = time.time() 
    y = pd_df['delinquency_12']
    t1 = time.time()
    print('\tselecting 1 column: %s seconds' % (t1 - t0))
    x = pd_df.drop(['delinquency_12'], axis=1)
    print('\tdropping 1 column: %s seconds' % (time.time() - t1))
    print('training_data_x.shape = ', x.shape)

    dtrain = xgb.DMatrix(x, y)
    t_dmatrix += time.time() - t0

    t0 = time.time()
    print('xgboost training temporarily disabled to save benchmarking time')
    model_xgb = None #xgb.train(dxgb_cpu_params, dtrain, num_boost_round=dxgb_cpu_params['nround'])
    t_train += time.time() - t0
    
    return model_xgb

# to download data for this script,
# visit https://rapidsai.github.io/demos/datasets/mortgage-data
# and update the following paths accordingly

if len(sys.argv) < 3:
    raise ValueError("needed to point path to mortgage folder, "
                     "count quarter to process")
else:
    mortgage_path = sys.argv[1]
    count_quarter_processing = int(sys.argv[2])
    #ml_fw = sys.argv[3]
    
acq_data_path = mortgage_path + "/acq"
perf_data_path = mortgage_path + "/perf"
col_names_path = mortgage_path + "/names.csv"

from pathlib import Path

def main():
    # end_year = 2016 # end_year is inclusive
    # part_count = 16 # the number of data files to train against
    # gpu_time = 0
    usePandas = '--use-pandas' in sys.argv

    socketCon = make_conn()
    db, socketCon = make_db(socketCon)

    shmemCon = make_conn(ipc=True)
    shDb, shmemCon = make_db(shmemCon, recreate=False)

    '''
    t0 = time.time()
    tmp1 = socketCon.create_table_from_csv('temp_perf', '/localdisk/vnlitvin/mortgage-fsi/perf/Performance_2000Q1.txt', ibis.Schema(
        names=("loan_id", "monthly_reporting_period", "servicer", "interest_rate", "current_actual_upb",
               "loan_age", "remaining_months_to_legal_maturity", "adj_remaining_months_to_maturity",
               "maturity_date", "msa", "current_loan_delinquency_status", "mod_flag", "zero_balance_code",
               "zero_balance_effective_date", "last_paid_installment_date", "foreclosed_after",
               "disposition_date", "foreclosure_costs", "prop_preservation_and_repair_costs",
               "asset_recovery_costs", "misc_holding_expenses", "holding_taxes", "net_sale_proceeds",
               "credit_enhancement_proceeds", "repurchase_make_whole_proceeds", "other_foreclosure_proceeds",
               "non_interest_bearing_upb", "principal_forgiveness_upb", "repurchase_make_whole_proceeds_flag",
               "foreclosure_principal_write_off_amount", "servicing_activity_indicator"),
        types=('int64', 'timestamp', 'category', 'float64', 'float64',
               'float64', 'float64', 'float64',
               'timestamp', 'float64', 'int32', 'category', 'category',
               'timestamp', 'timestamp', 'timestamp',
               'timestamp', 'float64', 'float64',
               'float64', 'float64', 'float64', 'float64',
               'float64', 'float64', 'float64',
               'float64', 'float64', 'category',
               'float64', 'category')
    ), DB_NAME)
    print('temp storage load: %s' % (time.time() - t0))

    return 0
    '''

    ACQ_SCHEMA = ibis.Schema(
        names=('loan_id', 'orig_channel', 'seller_name', 'orig_interest_rate', 'orig_upb', 'orig_loan_term',
               'orig_date', 'first_pay_date', 'orig_ltv', 'orig_cltv', 'num_borrowers', 'dti', 'borrower_credit_score',
               'first_home_buyer', 'loan_purpose', 'property_type', 'num_units', 'occupancy_status', 'property_state',
               'zip', 'mortgage_insurance_percent', 'product_type', 'coborrow_credit_score', 'mortgage_insurance_type',
               'relocation_mortgage_indicator', 'year_quarter_ignore'),
        types=('int64', 'category', 'string', 'float64', 'int64', 'int64',
               'timestamp', 'timestamp', 'float64', 'float64', 'float64', 'float64', 'float64',
               'category', 'category', 'category', 'int64', 'category', 'category',
               'int64', 'float64', 'category', 'float64', 'float64',
               'category', 'int32')
    )
    PERF_SCHEMA = ibis.Schema(
        names=("loan_id", "monthly_reporting_period", "servicer", "interest_rate", "current_actual_upb",
               "loan_age", "remaining_months_to_legal_maturity", "adj_remaining_months_to_maturity",
               "maturity_date", "msa", "current_loan_delinquency_status", "mod_flag", "zero_balance_code",
               "zero_balance_effective_date", "last_paid_installment_date", "foreclosed_after",
               "disposition_date", "foreclosure_costs", "prop_preservation_and_repair_costs",
               "asset_recovery_costs", "misc_holding_expenses", "holding_taxes", "net_sale_proceeds",
               "credit_enhancement_proceeds", "repurchase_make_whole_proceeds", "other_foreclosure_proceeds",
               "non_interest_bearing_upb", "principal_forgiveness_upb", "repurchase_make_whole_proceeds_flag",
               "foreclosure_principal_write_off_amount", "servicing_activity_indicator"),
        types=('int64', 'timestamp', 'category', 'float64', 'float64',
               'float64', 'float64', 'float64',
               'timestamp', 'float64', 'int32', 'category', 'category',
               'timestamp', 'timestamp', 'timestamp',
               'timestamp', 'float64', 'float64',
               'float64', 'float64', 'float64', 'float64',
               'float64', 'float64', 'float64',
               'float64', 'float64', 'category',
               'float64', 'category')
    )

    #names_table, con = make_table(con, DB_NAME, 'names', ibis.Schema(
    #    names=('seller_name', 'new'), types=('string', 'string')
    #))
    if usePandas:
        acq_table, con = make_table(socketCon, DB_NAME, 'acq', ACQ_SCHEMA)
        perf_table, con = make_table(socketCon, DB_NAME, 'perf', PERF_SCHEMA)


    global t_train, t_dmatrix

    pd_dfs = []
    perf_format_path = perf_data_path + "/Performance_%sQ%s.txt"

    time_ETL = time.time()
    for quarter in range(0, count_quarter_processing):
        year = 2000 + quarter // 4
        #perf_file = perf_format_path % (str(year), str(quarter % 4 + 1))

        files = [f for f in Path(perf_data_path).iterdir() if f.match('Performance_%sQ%s.txt*' % (str(year), str(quarter % 4 + 1)))]
        # print(files)
        for f in files:
            # print(f)
            if usePandas:
                acq_table, perf_table = import_data(socketCon, db, acq_table, perf_table, year=year, quarter=(quarter % 4 + 1), perf_file=str(f))
            #if not skipLoad:
            #    acq_table, perf_table = import_data(con, db, acq_table, perf_table, year=year, quarter=(quarter % 4 + 1), perf_file=str(f))
            else:
                with Timer('load acquisition'):
                    socketCon.create_table_from_csv('tmp_acq', '%s/Acquisition_%sQ%s.txt' % (acq_data_path, year, quarter % 4 + 1), ACQ_SCHEMA, DB_NAME)#, fragment_size=None)
                #    acq_table = socketCon.table('tmp_acq')
                #with Timer('transform acquisition'):
                #    socketCon.create_table('acq', acq_table, database=DB_NAME)
                #    socketCon.drop_table('tmp_acq')
                with Timer('load performance'):
                    socketCon.create_table_from_csv('tmp_perf', f, PERF_SCHEMA, DB_NAME)#, fragment_size=100000)
                #    perf_table = socketCon.table('tmp_perf')
                #with Timer('transform performance'):
                #    socketCon.create_table('perf', perf_table, database=DB_NAME)
                #    socketCon.drop_table('tmp_perf')
                

                acq_table = shDb.table('tmp_acq')
                perf_table = shDb.table('tmp_perf')

            pd_dfs.append(
                run_ibis_workflow(acq_table, perf_table)
            )
            if not usePandas:
                socketCon.drop_table('tmp_acq')
                socketCon.drop_table('tmp_perf')
            # print('finish f = ', f)

    time_ETL_end = time.time()

    print("ETL time: ", time_ETL_end - time_ETL)

    ##########################################################################
    print('pd_dfs.len = ', len(pd_dfs))

    #pd_df = pd_dfs[0]

    #one quarter at a time
    all_df = pd.concat(pd_dfs)

    print("concat df shape:", all_df.shape)

    #for pd_df in pd_dfs:
    #    ml_func(pd_df)

    train_xgb(all_df)    
    #train_xgb(pd_dfs[0])
    print('t_train = ', t_train)
    print('t_dmatrix = ', t_dmatrix)

    time_ML_train_end = time.time()
    print("Machine learning - train: ", time_ML_train_end - time_ETL_end)



if __name__ == '__main__':
    main()