# Derived from https://github.com/fschlimb/scale-out-benchs

import numpy as np
import pandas as pd
from pymapd import connect
from pandas.api.types import CategoricalDtype
from io import StringIO
from glob import glob
import os
import time
import pathlib
import sys
import argparse

def run_pd_workflow(quarter=1, year=2000, perf_file="", **kwargs):
    t1 = time.time()
    names = pd_load_names()
    year_string = str(year) + "Q" + str(quarter) + ".txt"
    acq_file = os.path.join(data_directory, "acq", "Acquisition_" + year_string)
    print("READING DATAFILE", acq_file)
    acq_pdf = pd_load_acquisition_csv(acq_file)

    print("READING DATAFILE", perf_file)
    perf_df_tmp = pd_load_performance_csv(perf_file)
    print("read time", (time.time() - t1) * 1000)

    t1 = time.time()

    acq_pdf = acq_pdf.merge(names, how='left', on=['seller_name'])
    acq_pdf.drop(columns=['seller_name'], inplace=True)
    acq_pdf['seller_name'] = acq_pdf['new']
    acq_pdf.drop(columns=['new'], inplace=True)

    pdf = perf_df_tmp
    everdf = create_ever_features(pdf)
    delinq_merge = create_delinq_features(pdf)
    everdf = join_ever_delinq_features(everdf, delinq_merge)
    del(delinq_merge)

    joined_df = create_joined_df(pdf, everdf)
    testdf = create_12_mon_features(joined_df)
    joined_df = combine_joined_12_mon(joined_df, testdf)
    del(testdf)

    perf_df = final_performance_delinquency(pdf, joined_df)
    del(pdf, joined_df)

    final_pdf = join_perf_acq_pdfs(perf_df, acq_pdf)
    del(perf_df)
    del(acq_pdf)

    print("compute time", (time.time() - t1) * 1000)
    final_pdf = last_mile_cleaning(final_pdf)
    exec_time = (time.time() - t1) * 1000
    print("compute time with copy to host", exec_time)
    return final_pdf, exec_time

def pd_load_performance_csv(performance_path, **kwargs):
    """ Loads performance data

    Returns
    -------
    PD DataFrame
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
    dtypes = {
        "loan_id": np.int64,
        "monthly_reporting_period": str,
        "servicer": str,
        "interest_rate": np.float64,
        "current_actual_upb": np.float64,
        "loan_age": np.float64,
        "remaining_months_to_legal_maturity": np.float64,
        "adj_remaining_months_to_maturity": np.float64,
        "maturity_date": str,
        "msa": np.float64,
        "current_loan_delinquency_status": np.int32,
        "mod_flag": CategoricalDtype(['N', 'Y']),
        "zero_balance_code": CategoricalDtype(['01', '02', '06', '09', '03', '15', '16']),
        "zero_balance_effective_date": str,
        "last_paid_installment_date": str,
        "foreclosed_after": str,
        "disposition_date": str,
        "foreclosure_costs": np.float64,
        "prop_preservation_and_repair_costs": np.float64,
        "asset_recovery_costs": np.float64,
        "misc_holding_expenses": np.float64,
        "holding_taxes": np.float64,
        "net_sale_proceeds": np.float64,
        "credit_enhancement_proceeds": np.float64,
        "repurchase_make_whole_proceeds": np.float64,
        "other_foreclosure_proceeds": np.float64,
        "non_interest_bearing_upb": np.float64,
        "principal_forgiveness_upb": np.float64,
        "repurchase_make_whole_proceeds_flag": CategoricalDtype(['N', 'Y']),
        "foreclosure_principal_write_off_amount": np.float64,
        "servicing_activity_indicator": CategoricalDtype(['N', 'Y']),
    }

    return pd.read_csv(performance_path, names=cols, delimiter='|', dtype=dtypes, parse_dates=[1,8,13,14,15,16])

def pd_load_acquisition_csv(acquisition_path, **kwargs):
    """ Loads acquisition data

    Returns
    -------
    PD DataFrame
    """

    columns = [
        'loan_id', 'orig_channel', 'seller_name', 'orig_interest_rate', 'orig_upb', 'orig_loan_term',
        'orig_date', 'first_pay_date', 'orig_ltv', 'orig_cltv', 'num_borrowers', 'dti', 'borrower_credit_score',
        'first_home_buyer', 'loan_purpose', 'property_type', 'num_units', 'occupancy_status', 'property_state',
        'zip', 'mortgage_insurance_percent', 'product_type', 'coborrow_credit_score', 'mortgage_insurance_type',
        'relocation_mortgage_indicator', 'year_quarter'
    ]
    dtypes = {
        'loan_id': np.int64,
        'orig_channel': CategoricalDtype(['B', 'C', 'R']),
        'seller_name': str,
        'orig_interest_rate': np.float64,
        'orig_upb': np.int64,
        'orig_loan_term': np.int64,
        'orig_date': str,
        'first_pay_date': str,
        'orig_ltv': np.float64,
        'orig_cltv': np.float64,
        'num_borrowers': np.float64,
        'dti': np.float64,
        'borrower_credit_score': np.float64,
        'first_home_buyer': CategoricalDtype(['N', 'U', 'Y']),
        'loan_purpose': CategoricalDtype(['C', 'P', 'R', 'U']),
        'property_type': CategoricalDtype(['CO', 'CP', 'MH', 'PU', 'SF']),
        'num_units': np.int64,
        'occupancy_status': CategoricalDtype(['I', 'P', 'S']),
        'property_state': CategoricalDtype(
            ['AK', 'AL', 'AR', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL', 'GA', 'HI',
            'IA', 'ID', 'IL', 'IN', 'KS', 'KY', 'LA', 'MA', 'MD', 'ME', 'MI', 'MN',
            'MO', 'MS', 'MT', 'NC', 'ND', 'NE', 'NH', 'NJ', 'NM', 'NV', 'NY', 'OH',
            'OK', 'OR', 'PA', 'PR', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VA', 'VI',
            'VT', 'WA', 'WI', 'WV', 'WY']),
        'zip': np.int64,
        'mortgage_insurance_percent': np.float64,
        'product_type': CategoricalDtype(['FRM']),
        'coborrow_credit_score': np.float64,
        'mortgage_insurance_type': np.float64,
        'relocation_mortgage_indicator': CategoricalDtype(['N', 'Y']),
        'year_quarter': np.int64
    }

    a = pd.read_csv(acquisition_path, names=columns, delimiter='|', dtype=dtypes, parse_dates=[6,7], error_bad_lines=True, warn_bad_lines=True, na_filter=True)
    return a

def pd_load_names(**kwargs):
    """ Loads names used for renaming the banks

    Returns
    -------
    PD DataFrame
    """

    cols = [
        'seller_name', 'new'
    ]

    dtypes = {'seller_name':str, 'new':str}

    return pd.read_csv(os.path.join(data_directory, "names.csv"), names=cols, delimiter='|', dtype=dtypes)

def create_ever_features(pdf, **kwargs):
    everdf = pdf[['loan_id', 'current_loan_delinquency_status']]
    everdf = everdf.groupby('loan_id').max()
    del(pdf)
    everdf['ever_30'] = (everdf['current_loan_delinquency_status'] >= 1).astype('int8')
    everdf['ever_90'] = (everdf['current_loan_delinquency_status'] >= 3).astype('int8')
    everdf['ever_180'] = (everdf['current_loan_delinquency_status'] >= 6).astype('int8')
    everdf.drop(columns=['current_loan_delinquency_status'], inplace=True)
    return everdf

def create_delinq_features(pdf, **kwargs):
    delinq_pdf = pdf[['loan_id', 'monthly_reporting_period', 'current_loan_delinquency_status']]
    del(pdf)
    delinq_30 = delinq_pdf.query('current_loan_delinquency_status >= 1')[['loan_id', 'monthly_reporting_period']].groupby('loan_id').min()
    delinq_30['delinquency_30'] = delinq_30['monthly_reporting_period']
    delinq_30.drop(columns=['monthly_reporting_period'], inplace=True)
    delinq_90 = delinq_pdf.query('current_loan_delinquency_status >= 3')[['loan_id', 'monthly_reporting_period']].groupby('loan_id').min()
    delinq_90['delinquency_90'] = delinq_90['monthly_reporting_period']
    delinq_90.drop(columns=['monthly_reporting_period'], inplace=True)
    delinq_180 = delinq_pdf.query('current_loan_delinquency_status >= 6')[['loan_id', 'monthly_reporting_period']].groupby('loan_id').min()
    delinq_180['delinquency_180'] = delinq_180['monthly_reporting_period']
    delinq_180.drop(columns=['monthly_reporting_period'], inplace=True)
    del(delinq_pdf)
    delinq_merge = delinq_30.merge(delinq_90, how='left', on=['loan_id'])
    delinq_merge['delinquency_90'] = delinq_merge['delinquency_90'].fillna(np.dtype('datetime64[ms]').type('1970-01-01').astype('datetime64[ms]'))
    delinq_merge = delinq_merge.merge(delinq_180, how='left', on=['loan_id'])
    delinq_merge['delinquency_180'] = delinq_merge['delinquency_180'].fillna(np.dtype('datetime64[ms]').type('1970-01-01').astype('datetime64[ms]'))
    del(delinq_30)
    del(delinq_90)
    del(delinq_180)
    return delinq_merge

def join_ever_delinq_features(everdf_tmp, delinq_merge, **kwargs):
    everdf = everdf_tmp.merge(delinq_merge, on=['loan_id'], how='left')
    del(everdf_tmp)
    del(delinq_merge)
    everdf['delinquency_30'] = everdf['delinquency_30'].fillna(np.dtype('datetime64[ms]').type('1970-01-01').astype('datetime64[ms]'))
    everdf['delinquency_90'] = everdf['delinquency_90'].fillna(np.dtype('datetime64[ms]').type('1970-01-01').astype('datetime64[ms]'))
    everdf['delinquency_180'] = everdf['delinquency_180'].fillna(np.dtype('datetime64[ms]').type('1970-01-01').astype('datetime64[ms]'))
    return everdf

def create_joined_df(pdf, everdf, **kwargs):
    test = pdf[['loan_id', 'monthly_reporting_period', 'current_loan_delinquency_status', 'current_actual_upb']]
    del(pdf)
    test['timestamp'] = test['monthly_reporting_period']
    test.drop(columns=['monthly_reporting_period'], inplace=True)
    test['timestamp_month'] = test['timestamp'].dt.month
    test['timestamp_year'] = test['timestamp'].dt.year
    test['delinquency_12'] = test['current_loan_delinquency_status']
    test.drop(columns=['current_loan_delinquency_status'], inplace=True)
    test['upb_12'] = test['current_actual_upb']
    test.drop(columns=['current_actual_upb'], inplace=True)
    test['upb_12'] = test['upb_12'].fillna(999999999)
    test['delinquency_12'] = test['delinquency_12'].fillna(-1)

    joined_df = test.merge(everdf, how='left', on=['loan_id'])
    del(everdf)
    del(test)

    joined_df['ever_30'] = joined_df['ever_30'].fillna(-1)
    joined_df['ever_90'] = joined_df['ever_90'].fillna(-1)
    joined_df['ever_180'] = joined_df['ever_180'].fillna(-1)
    joined_df['delinquency_30'] = joined_df['delinquency_30'].fillna(-1)
    joined_df['delinquency_90'] = joined_df['delinquency_90'].fillna(-1)
    joined_df['delinquency_180'] = joined_df['delinquency_180'].fillna(-1)

    joined_df['timestamp_year'] = joined_df['timestamp_year'].astype('int32')
    joined_df['timestamp_month'] = joined_df['timestamp_month'].astype('int32')

    return joined_df

def create_12_mon_features(joined_df, **kwargs):
    testdfs = []
    n_months = 12
    for y in range(1, n_months + 1):
        tmpdf = joined_df[['loan_id', 'timestamp_year', 'timestamp_month', 'delinquency_12', 'upb_12']]
        tmpdf['josh_months'] = tmpdf['timestamp_year'] * 12 + tmpdf['timestamp_month']
        tmpdf['josh_mody_n'] = np.floor((tmpdf['josh_months'].astype('float64') - 24000 - y) / 12)
        tmpdf = tmpdf.groupby(['loan_id', 'josh_mody_n'], as_index=False).agg({'delinquency_12': 'max','upb_12': 'min'})
        tmpdf['delinquency_12'] = (tmpdf['delinquency_12']>3).astype('int32')
        tmpdf['delinquency_12'] +=(tmpdf['upb_12']==0).astype('int32')
        tmpdf['timestamp_year'] = np.floor(((tmpdf['josh_mody_n'] * n_months) + 24000 + (y - 1)) / 12).astype('int16')
        tmpdf['timestamp_month'] = np.int8(y)
        tmpdf.drop(columns=['josh_mody_n'], inplace=True)
        testdfs.append(tmpdf)
        del(tmpdf)
    del(joined_df)

    return pd.concat(testdfs)

def combine_joined_12_mon(joined_df, testdf, **kwargs):
    joined_df.drop(columns=['delinquency_12', 'upb_12'], inplace=True)
    joined_df['timestamp_year'] = joined_df['timestamp_year'].astype('int16')
    joined_df['timestamp_month'] = joined_df['timestamp_month'].astype('int8')
    return joined_df.merge(testdf, how='left', on=['loan_id', 'timestamp_year', 'timestamp_month'])

def final_performance_delinquency(merged, joined_df, **kwargs):
    merged['timestamp_month'] = merged['monthly_reporting_period'].dt.month
    merged['timestamp_month'] = merged['timestamp_month'].astype('int8')
    merged['timestamp_year'] = merged['monthly_reporting_period'].dt.year
    merged['timestamp_year'] = merged['timestamp_year'].astype('int16')
    merged = merged.merge(joined_df, how='left', on=['loan_id', 'timestamp_year', 'timestamp_month'])
    merged.drop(columns=['timestamp_year'], inplace=True)
    merged.drop(columns=['timestamp_month'], inplace=True)
    return merged

def join_perf_acq_pdfs(perf, acq, **kwargs):
    return perf.merge(acq, how='left', on=['loan_id'])

def last_mile_cleaning(df, **kwargs):
    #for col, dtype in df.dtypes.iteritems():
    #    if str(dtype)=='category':
    #        df[col] = df[col].cat.codes
    df['delinquency_12'] = df['delinquency_12'] > 0
    df['delinquency_12'] = df['delinquency_12'].fillna(False).astype('int32')
    return df #.to_arrow(index=False)

# Load database reporting functions
pathToReportDir = os.path.join(pathlib.Path(__file__).parent, "..", "report")
print(pathToReportDir)
sys.path.insert(1, pathToReportDir)
import report

parser = argparse.ArgumentParser(description='Run Mortgage benchmark using pandas')

parser.add_argument('-r', default="report_pandas.csv", help="Report file name.")
parser.add_argument('-df', default=1, type=int, help="Number of datafiles (quarters) to input into database for processing.")
parser.add_argument('-dp', required=True, help="Path to root of mortgage datafiles directory (contains names.csv).")
parser.add_argument('-i', dest="iterations", default=5, type=int, help="Number of iterations to run every benchmark. Best result is selected.")

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

if args.iterations < 1:
    print("Bad number of iterations specified", args.t)

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
        'ScriptName': 'mortgage_pandas.py',
        'CommitHash': args.commit
    })

data_directory = args.dp
benchName = "mortgage_pandas"

perf_data_path = os.path.join(data_directory, "perf")
perf_format_path = os.path.join(perf_data_path, "Performance_%sQ%s.txt")
bestExecTime = float("inf")
bestTotalTime = float("inf")
worstExecTime = 0
worstTotalTime = 0
avgExecTime = 0
avgTotalTime = 0

for iii in range(1, args.iterations + 1):
    dataFilesNumber = 0
    time_ETL = time.time()
    exec_time_total = 0
    print("RUNNING BENCHMARK NUMBER", benchName, "ITERATION NUMBER", iii)
    for quarter in range(0, args.df):
        year = 2000 + quarter // 4
        perf_file = perf_format_path % (str(year), str(quarter % 4 + 1))

        files = [f for f in pathlib.Path(perf_data_path).iterdir() if f.match('Performance_%sQ%s.txt*' % (str(year), str(quarter % 4 + 1)))]
        for f in files:
            dataframe, exec_time = run_pd_workflow(year = year, quarter = (quarter % 4 + 1), perf_file = str(f))
            exec_time_total += exec_time
        dataFilesNumber += 1
    time_ETL_end = time.time()
    ttt = (time_ETL_end - time_ETL) * 1000
    print("ITERATION", iii, "EXEC TIME: ", exec_time_total, "TOTAL TIME: ", ttt)

    if bestExecTime > exec_time_total:
        bestExecTime = exec_time_total
    if worstExecTime < exec_time_total:
        worstExecTime = exec_time_total
    avgExecTime += exec_time_total
    if bestTotalTime > ttt:
        bestTotalTime = ttt
    if worstTotalTime < ttt:
        bestTotalTime = ttt
    avgTotalTime += ttt

avgExecTime /= args.iterations
avgTotalTime /= args.iterations

try:
    with open(args.r, "w") as report:
        print("BENCHMARK", benchName, "EXEC TIME", bestExecTime, "TOTAL TIME", bestTotalTime)
        print("datafiles,fragment_size,query,query_exec_min,query_total_min,query_exec_max,query_total_max,query_exec_avg,query_total_avg,query_error_info", file=report, flush=True)
        print(dataFilesNumber, ",",
              0, ",",
              benchName, ",",
              bestExecTime, ",",
              bestTotalTime, ",",
              worstExecTime, ",",
              worstTotalTime, ",",
              avgExecTime, ",",
              avgTotalTime, ",",
              "", '\n', file=report, sep='', end='', flush=True)
        if db_reporter is not None:
            db_reporter.submit({
                'FilesNumber': dataFilesNumber,
                'FragmentSize': 0,
                'BenchName': benchName,
                'BestExecTimeMS': bestExecTime,
                'BestTotalTimeMS': bestTotalTime,
                'WorstExecTimeMS': worstExecTime,
                'WorstTotalTimeMS': worstTotalTime,
                'AverageExecTimeMS': avgExecTime,
                'AverageTotalTimeMS': avgTotalTime})
except IOError as err:
    print("Failed writing report file", args.r, err)
