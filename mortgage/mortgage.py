# Derived from https://github.com/fschlimb/scale-out-benchs

import numpy as np
import pandas as pd
from pymapd import connect
from pandas.api.types import CategoricalDtype
from io import StringIO
from glob import glob
import os
import time
#import cudf, io, requests
data_directory= "/nfs/site/proj/scripting_tools/gashiman/mortgage-data"
con = connect(user="admin", password="HyperInteractive", host="localhost", dbname="omnisci", port=62740)
#mapd_con =mapd_jdbc.connect(user="admin", password="HyperInteractive", host="localhost", dbname="omnisci", port=62740)

def run_pd_workflow(quarter=1, year=2000, perf_file="", **kwargs):
   # con = connect(user="admin", password="HyperInteractive", host="localhost", dbname="omnisci", port=62740)
    t1 = time.time()
    # Load names
    names = pd_load_names()
    con.execute('DROP TABLE IF EXISTS names;')
    con.create_table('names', names)
    # Load acquisition
    acq_pdf = pd_load_acquisition_csv(acquisition_path =
            os.path.join(data_directory, "acq", "Acquisition_" + str(year) + "Q" + str(quarter) + ".txt"))
    con.execute('DROP TABLE IF EXISTS acq;')
    con.create_table('acq', acq_pdf)
    # Load perf
    perf_df_tmp = pd_load_performance_csv(perf_file)
    con.execute('DROP TABLE IF EXISTS perf;')
    con.create_table('perf', perf_df_tmp)
    print("read time", time.time()-t1)

    t1 = time.time()
    con.execute('DROP TABLE IF EXISTS acqtemp;');
    con.execute('CREATE TABLE acqtemp AS SELECT loan_id,orig_channel,year_quarter,names.seller_name AS seller_name,new_seller_name FROM acq  LEFT JOIN names ON acq.seller_name = names.seller_name;');
    con.execute('DROP TABLE IF EXISTS acq;');
    con.execute('ALTER TABLE acqtemp RENAME TO acq;');
    con.execute('DROP TABLE IF EXISTS names;');
     #acq_pdf = acq_pdf.merge(names, how='left', on=['seller_name'])
    #acq_pdf.drop(columns=['seller_name'], inplace=True)
    # acq_pdf['seller_name'] = acq_pdf['new_seller_name']
    #acq_pdf.drop(columns=['new_seller_name'], inplace=True)
    # DECLARE @pdf nvarchar(30)
    #SET pdf = perf_df_tmp
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
    print("compute time", time.time()-t1)
    final_pdf = last_mile_cleaning(final_pdf)
    print("compute time with copy to host", time.time()-t1)
    return final_pdf

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

    print(performance_path)

    p = pd.read_csv(performance_path, names=cols, delimiter='|', dtype=dtypes, parse_dates=[1,8,13,14,15,16])
    print(p.info())
    return p

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

    print(acquisition_path)

    a = pd.read_csv(acquisition_path, names=columns, delimiter='|', dtype=dtypes, parse_dates=[6,7], error_bad_lines=True, warn_bad_lines=True, na_filter=True)
    print (a.info())
    return a

def pd_load_names(**kwargs):
    """ Loads names used for renaming the banks

     Returns
    -------
    PD DataFrame
    """

    cols = [
        'seller_name', 'new_seller_name'
    ]

    dtypes = {'seller_name':str, 'new_seller_name':str}

    n = pd.read_csv(os.path.join(data_directory, "names.csv"), names=cols, delimiter='|', dtype=dtypes)
    print (n.info())
    return n

def create_ever_features(pdf, **kwargs):
    everdf = pdf[['loan_id', 'current_loan_delinquency_status']]
    con.execute('DROP TABLE IF EXISTS everdf;');
    con.execute('DROP TABLE IF EXISTS everdftemp1;');
    con.create_table('everdf', everdf);
    con.execute('CREATE TABLE everdftemp1 AS select MAX(loan_id) AS load_id,everdf.current_loan_delinquency_status AS current_loan_delinquency_status FROM everdf group by loan_id,current_loan_delinquency_status;');
    #everdf = everdf.groupby('loan_id').max()
    con.execute('DROP TABLE IF EXISTS pdf;');
    #del(pdf)
     #con.execute('ALTER TABLE everdftemp1 ADD (ever_30 INT   ENCODING DICT);');
    con.execute('DROP TABLE IF EXISTS ever_30;');
    con.execute('DROP TABLE IF EXISTS ever_90;');
    con.execute('DROP TABLE IF EXISTS ever_180;');
    con.execute('DROP TABLE IF EXISTS ever_30temp;');
    con.execute('DROP TABLE IF EXISTS ever_90temp;');
    con.execute('DROP TABLE IF EXISTS ever_180temp;');
    #con.execute()
    con.execute('CREATE TABLE ever_30 AS SELECT loan_id,current_loan_delinquency_status FROM everdf where current_loan_delinquency_status >= 1;');
    con.execute('UPDATE ever_30 SET current_loan_delinquency_status = CAST(current_loan_delinquency_status AS BIGINT);');
    con.execute('CREATE TABLE ever_90 AS SELECT loan_id,current_loan_delinquency_status FROM everdf where current_loan_delinquency_status >= 3;');
    con.execute('UPDATE ever_90 SET current_loan_delinquency_status = CAST(current_loan_delinquency_status AS BIGINT);')    ;
    con.execute('CREATE TABLE ever_180 AS SELECT loan_id,current_loan_delinquency_status FROM everdf where current_loan_delinquency_status >= 6;');
    con.execute('UPDATE ever_180 SET current_loan_delinquency_status = CAST(current_loan_delinquency_status AS BIGINT);')    ;
    con.execute('CREATE TABLE ever_30temp AS SELECT everdftemp1.load_id AS loan_id,everdftemp1.current_loan_delinquency_status AS current_loan_delinquency_status ,ever_30.current_loan_delinquency_status AS ever_30 FROM ever_30  LEFT JOIN everdftemp1 ON ever_30.loan_id = everdftemp1.load_id;');
    con.execute('CREATE TABLE ever_90temp AS SELECT ever_30temp.loan_id AS loan_id,ever_30temp.current_loan_delinquency_status AS current_loan_delinquency_status,ever_30temp.current_loan_delinquency_status AS ever_30 ,ever_90.current_loan_delinquency_status AS ever_90 FROM ever_30temp  LEFT JOIN ever_90 ON ever_90.loan_id = ever_30temp.loan_id;');
    con.execute('CREATE TABLE ever_180temp AS SELECT ever_90temp.loan_id AS loan_id,ever_90temp.current_loan_delinquency_status AS current_loan_delinquency_status,ever_90temp.ever_30 AS ever_30,ever_90temp.ever_90 ,ever_180.current_loan_delinquency_status AS ever_180 FROM ever_90temp  LEFT JOIN ever_180 ON ever_180.loan_id = ever_90temp.loan_id;');
    con.execute('DROP TABLE IF EXISTS everdftemp1;');
    con.execute('DROP TABLE IF EXISTS mergetest;');
    #con.execute('CREATE TABLE mergetest');
    #con.execute('ALTER TABLE ever_30 ADD COLUMN test INT;');
    #con.execute('INSERT INTO ever_30 (test) select current_loan_delinquency_status FROM everdf;');
    con.execute('DROP TABLE IF EXISTS everdf1;');
   # con.execute('DROP TABLE IF EXISTS acq;');
    con.execute('ALTER TABLE ever_180temp  RENAME TO everdf1;');
     #con.execute('ALTER TABLE everdftemp1 RENAME TO everdf;');
    #con.execute('CREATE TABLE everdftemp1 AS SELECT loan_id FROM everdf;');
    #con.execute('CREATE TABLE everdftemp1 AS SELECT loan_id FROM everdf;');
    #con.execute('CREATE TABLE everdftemp1 AS SELECT loan_id FROM everdf;');
    #con.execute('DROP TABLE IF EXISTS everdf;');
   # con.execute('ALTER TABLE ever_180temp  RENAME TO everdftemp1 ;');
    con.execute('DROP TABLE IF EXISTS ever_30;');
    con.execute('DROP TABLE IF EXISTS ever_90;');
    con.execute('DROP TABLE IF EXISTS ever_180;');
    con.execute('DROP TABLE IF EXISTS ever_30temp;');
    con.execute('DROP TABLE IF EXISTS ever_90temp;');
    con.execute('DROP TABLE IF EXISTS ever_180temp;');
    #everdf.drop(columns=['current_loan_delinquency_status'], inplace=True)
    return everdf


def create_delinq_features(pdf, **kwargs):
    delinq_pdf = pdf[['loan_id', 'monthly_reporting_period', 'current_loan_delinquency_status']]
    con.execute('DROP TABLE IF EXISTS deling;');
    con.execute('DROP TABLE IF EXISTS deling_30 ;');
    con.execute('DROP TABLE IF EXISTS deling_90 ;');
    con.execute('DROP TABLE IF EXISTS deling_180 ;');

    con.create_table('deling',pdf);
    del(pdf)
     #delinq_30 = delinq_pdf.query('current_loan_delinquency_status >= 1')[['loan_id', 'monthly_reporting_period']].groupby('loan_id').min()
    con.execute('CREATE TABLE  deling_30 AS (SELECT MIN(loan_id) AS load_id ,monthly_reporting_period AS delinquency_30 FROM deling where current_loan_delinquency_status >= 1 GROUP BY loan_id,monthly_reporting_period);');
   # delinq_30['delinquency_30'] = delinq_30['monthly_reporting_period']
    con.execute('CREATE TABLE  deling_90 AS SELECT MIN(loan_id) AS load_id,monthly_reporting_period AS delinquency_90 FROM deling where current_loan_delinquency_status >= 3 GROUP BY loan_id,monthly_reporting_period;');
    con.execute('CREATE TABLE  deling_180 AS SELECT MIN(loan_id) AS load_id,monthly_reporting_period AS delinquency_180 FROM deling where current_loan_delinquency_status >= 6 GROUP BY loan_id,monthly_reporting_period;');

    # delinq_90 = delinq_pdf.query('current_loan_delinquency_status >= 3')[['loan_id', 'monthly_reporting_period']].groupby('loan_id').min()
   # delinq_90['delinquency_90'] = delinq_90['monthly_reporting_period']
    #delinq_90.drop(columns=['monthly_reporting_period'], inplace=True)
    #delinq_180 = delinq_pdf.query('current_loan_delinquency_status >= 6')[['loan_id', 'monthly_reporting_period']].groupby('loan_id').min()
    #delinq_180['delinquency_180'] = delinq_180['monthly_reporting_period']
    #delinq_180.drop(columns=['monthly_reporting_period'], inplace=True)
    del(delinq_pdf)
    con.execute('DROP TABLE IF EXISTS deling_merge;')
    con.execute('CREATE TABLE deling_merge AS SELECT deling_30.delinquency_30 AS delinquency_30 ,deling_30.load_id AS loan_id,deling_90.delinquency_90 AS delinquency_90 FROM deling_30  LEFT JOIN deling_90 ON deling_30.load_id= deling_90.load_id;');
    # delinq_merge = delinq_30.merge(delinq_90, how='left', on=['loan_id'])
    #UPDATE UFOs SET shape='ovate' where shape='eggish'; 23:59:59.999
    #con.execute(' EXTRACT MONTH FROM 2018-08-01;');
    con.execute('UPDATE deling_merge SET  delinquency_90 = NULL where  delinquency_90 = NULL');
    # delinq_merge['delinquency_90'] = delinq_merge['delinquency_90'].fillna(np.dtype('datetime64[ms]').type('1970-01-01').astype('datetime64[ms]'))
    con.execute('DROP TABLE IF EXISTS deling_mergetemp;');
    con.execute('CREATE TABLE deling_mergetemp AS SELECT deling_merge.delinquency_30 AS delinquency_30 ,deling_merge.loan_id AS loan_id ,deling_merge.delinquency_90 AS delinquency_90,deling_180.delinquency_180 AS delinquency_180  FROM deling_merge  LEFT JOIN deling_180 ON deling_merge.loan_id= deling_180.load_id;');
    # con.execute('UPDATE deling_merge SET  delinquency_90 = NULL where  delinquency_90 = NULL');
    #delinq_merge = delinq_merge.merge(delinq_180, how='left', on=['loan_id'])
    con.execute('DROP TABLE IF EXISTS deling_merge;');
    con.execute('ALTER TABLE deling_mergetemp RENAME TO deling_merge;');
    #t =('test1;');
    #con.execute('UPDATE deling_merge SET  delinquency_180 = NULL where  delinquency_180 = NULL');
    #print(row=con.execute("SELECT * from deling_merge"));
    #rows=con.fetchall();
    #delinq_merge['delinquency_180'] = delinq_merge['delinquency_180'].fillna(np.dtype('datetime64[ms]').type('1970-01-01').astype('datetime64[ms]'))
    #del(delinq_30)
    #del(delinq_90)
    #del(delinq_180)
    con.execute('DROP TABLE IF EXISTS deling_30;');
    con.execute('DROP TABLE IF EXISTS deling_90;');
    con.execute('DROP TABLE IF EXISTS deling_180;');
    mapd_cursor = con.cursor();
    #con_cursor=con.cursor();
    row="SELECT delinquency_30,loan_id,delinquency_90,delinquency_180 FROM deling_merge";
    mapd_cursor.execute(row);
    deling_merge_temp = mapd_cursor.fetchall();
    deling_merge = pd.DataFrame(deling_merge_temp);
    d =deling_merge.head();
    return deling_merge


def join_ever_delinq_features(everdf_tmp, delinq_merge, **kwargs):
    #everdf = everdf_tmp.merge(delinq_merge, on=['loan_id'], how='left')
    con.execute('DROP TABLE IF EXISTS ever;')
    con.execute('DROP TABLE IF EXISTS everdf;')
    con.create_table('ever', everdf_tmp)
    #con.execute('DRPO TABLE IF EXISTS everdf;');
    #con.create_table('everdf',everdf);
    con.execute('CREATE TABLE everdf AS SELECT deling_merge.delinquency_30 AS delinquency_30 ,deling_merge.loan_id as Loan_id,deling_merge.delinquency_90 AS delinquency_90,deling_merge.delinquency_180 AS delinquency_180  FROM deling_merge  LEFT JOIN ever ON deling_merge.loan_id= ever.loan_id;');
    #con.execute('UPDATE everdf SET delinquency_90 =CAST (01/01/1970 AS TIMESTAMP(0)) where delinquency_90 = NULL;');
    del(everdf_tmp)
    del(delinq_merge)
    #everdf['delinquency_30'] = everdf['delinquency_30'].fillna(np.dtype('datetime64[ms]').type('1970-01-01').astype('datetime64[ms]'))
    #everdf['delinquency_90'] = everdf['delinquency_90'].fillna(np.dtype('datetime64[ms]').type('1970-01-01').astype('datetime64[ms]'))
    #everdf['delinquency_180'] = everdf['delinquency_180'].fillna(np.dtype('datetime64[ms]').type('1970-01-01').astype('datetime64[ms]'))
    mapd_cursor = con.cursor();
    row1="SELECT delinquency_30,loan_id,delinquency_90,delinquency_180 FROM everdf";
    mapd_cursor.execute(row1);
    deling_merge_temp1 = mapd_cursor.fetchall();
    everdf = pd.DataFrame(deling_merge_temp1);

    return everdf

def create_joined_df(pdf, everdf, **kwargs):
    test = pdf[['loan_id', 'monthly_reporting_period', 'current_loan_delinquency_status', 'current_actual_upb']]
    con.execute('DROP TABLE IF EXISTS test;')
    con.create_table('test', test)
    #con.execute('DROP TABLE IF EXISTS mtemp;')
    #con.execute('DROP TABLE IF EXISTS ytemp;')
    #con.execute('DROP TABLE IF EXISTS yeartemp;')
    #con.execute('DROP TABLE IF EXISTS monthtemp;')
    con.execute('DROP TABLE IF EXISTS TEST2;')
    #del(pdf)
    con.execute('CREATE TABLE TEST2 AS SELECT  EXTRACT (YEAR FROM monthly_reporting_period) timestamp_year,EXTRACT (MONTH FROM monthly_reporting_period) timestamp_month, loan_id,current_loan_delinquency_status AS delinquency_12,monthly_reporting_period AS timestamp_temp,current_actual_upb AS upb_12  FROM test');
    con.execute('DROP TABLE IF EXISTS test;')
    con.execute('ALTER TABLE TEST2  RENAME TO test;');
    con.execute('UPDATE test SET upb_12 = 999999999 where upb_12 = NULL;');
    con.execute('UPDATE test SET delinquency_12 = -1  where delinquency_12 = NULL;');
   # test['timestamp'] = test['monthly_reporting_period']
   # test.drop(columns=['monthly_reporting_period'], inplace=True)
    #test['timestamp_month'] = test['timestamp'].dt.month
    #test['timestamp_year'] = test['timestamp'].dt.year
    #test['delinquency_12'] = test['current_loan_delinquency_status']
    #test.drop(columns=['current_loan_delinquency_status'], inplace=True)
    #test['upb_12'] = test['current_actual_upb']
    #test.drop(columns=['current_actual_upb'], inplace=True)
    # test['upb_12'] = test['upb_12'].fillna(999999999)
    # test['delinquency_12'] = test['delinquency_12'].fillna(-1)
    con.execute('DROP TABLE IF EXISTS joined_df;')
    con.execute('DROP TABLE IF EXISTS joined_df1;')

    con.execute('CREATE TABLE joined_df AS SELECT test.loan_id AS loan_id,test.timestamp_temp AS timestamp_temp,test.delinquency_12 AS delinquency_12, test.upb_12 AS upb_12,test.timestamp_year AS timestamp_year ,test.timestamp_month AS timestamp_month,everdf.delinquency_30 AS delinquency_30 ,everdf.delinquency_90 AS delinquency_90,everdf.delinquency_180 AS delinquency_180 FROM test LEFT JOIN everdf  ON test.loan_id= everdf.loan_id;');
    con.execute('CREATE TABLE joined_df1 AS SELECT joined_df.loan_id AS loan_id,joined_df.timestamp_temp AS timestamp_temp,joined_df.delinquency_12 AS delinquency_12,joined_df.upb_12 AS upb_12,joined_df.timestamp_year AS timestamp_year ,joined_df.timestamp_month AS timestamp_month,joined_df.delinquency_30 AS delinquency_30 ,joined_df.delinquency_90 AS delinquency_90,joined_df.delinquency_180 AS delinquency_180,everdf1.ever_30 AS ever_30,everdf1.ever_90 AS ever_90,everdf1.ever_180 AS ever_180 FROM joined_df LEFT JOIN everdf1  ON joined_df.loan_id= everdf1.loan_id;');

    # con.execute('DROP TABLE IF EXISTS everdf;')
    con.execute('DROP TABLE IF EXISTS test;')
    con.execute('DROP TABLE IF EXISTS joined_df;')
    con.execute('ALTER TABLE joined_df1 RENAME TO joined_df;')
    #del(everdf)
    #del(test)
    con.execute('UPDATE joined_df SET ever_30 = -1  where ever_30 = NULL;');
    con.execute('UPDATE joined_df SET ever_90 = -1  where ever_90 = NULL;');
    con.execute('UPDATE joined_df SET ever_180 = -1  where ever_180 = NULL;');
    #con.execute('UPDATE joined_df SET delinquency_90 = -1  where delinquency_90 = NULL;');
    #con.execute('UPDATE joined_df SET delinquency_180 = -1  where delinquency_180 = NULL;');
   # joined_df['ever_30'] = joined_df['ever_30'].fillna(-1)
   # joined_df['ever_90'] = joined_df['ever_90'].fillna(-1)
   # joined_df['ever_180'] = joined_df['ever_180'].fillna(-1)
   # joined_df['delinquency_30'] = joined_df['delinquency_30'].fillna(-1)
   # joined_df['delinquency_90'] = joined_df['delinquency_90'].fillna(-1)
   # joined_df['delinquency_180'] = joined_df['delinquency_180'].fillna(-1)
    con.execute('UPDATE joined_df SET upb_12 = 999999999 where upb_12 = NULL;');
    #Ijoined_df['timestamp_year'] = joined_df['timestamp_year'].astype('int32')
    #joined_df['timestamp_month'] = joined_df['timestamp_month'].astype('int32')

    #return joined_df
    #mapd_cursor =con.cursor()
    #query="SELECT loan_id, timestamp_temp,delinquency_12,upb_12,timestamp_year ,timestamp_month,delinquency_30 ,delinquency_90,delinquency_180,ever_30,ever_90,ever_180 from joined_df"
    #mapd_cursor.execute(query)
    #result=mapd_cursor.fetchall()
    #df =pd.DataFrame(result)
    #joined_df =df
    # df = con.select_ipc_gpu(query)
     #df =con.select_ipc_gpu(query)
     #df.head()
    #print("hahaaa")
    #print(result)
    #print(df)
    #return joined_df

def create_12_mon_features(joined_df, **kwargs):
    testdfs = []
    n_months = 12
    for y in range(1, n_months + 1):
     string =str(y)
     months_string=str(n_months)
     #print(string)
     con.execute('DROP TABLE IF EXISTS tmpdf;')
    # con.execute('DROP TABLE IF EXISTS josh_monthstemp;')
     con.execute('DROP TABLE IF EXISTS deling_12 ;');
     #con.execute('DROP TABLE IF EXISTS josh_monthsjoin ;');
     con.execute('DROP TABLE IF EXISTS josh_mody_ntemp ;');
     con.execute('DROP TABLE IF EXISTS finaltbl ;');
     # con.execute('SET string = y ;');
     con.execute('CREATE TABLE tmpdf AS SELECT loan_id,timestamp_year,((timestamp_year * 12)+timestamp_month) AS josh_months,timestamp_month,delinquency_12,upb_12 from joined_df;')
       #tmpdf = joined_df[['loan_id', 'timestamp_year', 'timestamp_month', 'delinquency_12', 'upb_12']]
     #con.execute('CREATE TABLE josh_monthstemp AS SELECT ((timestamp_year * 12)+timestamp_month) AS josh_months,loan_id FROM tmpdf;');
     #con.execute('CREATE TABLE josh_monthsjoin  AS SELECT tmpdf.loan_id AS loan_id, tmpdf.timestamp_year AS timestamp_year ,tmpdf.timestamp_month AS timestamp_month,tmpdf.delinquency_12 AS delinquency_12,tmpdf.upb_12 AS upb_12,josh_monthstemp.josh_months AS josh_months FROM tmpdf  LEFT JOIN josh_monthstemp  ON josh_monthstemp.loan_id= tmpdf.loan_id;');

  #tmpdf['josh_months'] = tmpdf['timestamp_year'] * 12 + tmpdf['timestamp_month']
     con.execute('CREATE TABLE deling_12 AS SELECT FLOOR((josh_months-24000-'+string+')/12) AS josh_mody_n,loan_id from tmpdf;');
     con.execute('CREATE TABLE josh_mody_ntemp  AS SELECT tmpdf.timestamp_year AS timestamp_year ,tmpdf.timestamp_month AS timestamp_month,((delinquency_12 > 3) AND (upb_12 =0)) AS delinquency_12,tmpdf.upb_12 AS upb_12,tmpdf.josh_months AS josh_months ,tmpdf.loan_id AS loan_id, deling_12.josh_mody_n AS josh_mody_n FROM tmpdf  LEFT JOIN deling_12  ON tmpdf.loan_id= deling_12.loan_id;');

       #tmpdf['josh_mody_n'] = np.floor((tmpdf['josh_months'].astype('float64') - 24000 - y) / 12)

  #tmpdf = tmpdf.groupby(['loan_id', 'josh_mody_n'], as_index=False).agg({'delinquency_12': 'max','upb_12': 'min'})
    # con.execute('DROP TABLE IF EXISTS deling_12temp;')
     con.execute('DROP TABLE IF EXISTS timestamp_yeartemp;')
    # con.execute('DROP TABLE IF EXISTS deling_12_two;')
    # con.execute('CREATE TABLE deling_12temp AS SELECT ((delinquency_12 >3) AND (upb_12 = 0)) AS delinquency_12 ,loan_id FROM tmpdf;');
    # con.execute('CREATE TABLE deling_12_two  AS SELECT josh_mody_ntemp.timestamp_year AS timestamp_year ,josh_mody_ntemp.timestamp_month AS timestamp_month,josh_mody_ntemp.delinquency_12 AS delinquency_12,josh_mody_ntemp.upb_12 AS upb_12,josh_mody_ntemp.josh_months AS josh_months,josh_mody_ntemp.loan_id AS loan_id ,josh_mody_ntemp.josh_mody_n AS josh_mody_n,deling_12temp.delinquency_12 AS delinquency_12 FROM josh_mody_ntemp  LEFT JOIN deling_12temp  ON josh_mody_ntemp.loan_id= deling_12temp.loan_id;');
     #tmpdf['delinquency_12'] = (tmpdf['delinquency_12']>3).astype('int32')
     #tmpdf['delinquency_12'] +=(tmpdf['upb_12']==0).astype('int32')
     #tmpdf['timestamp_year'] = np.floor(((tmpdf['josh_mody_n'] * n_months) + 24000 + (y - 1)) / 12).astype('int16')
     con.execute('CREATE TABLE timestamp_yeartemp AS SELECT ((josh_mody_n * '+months_string+') +2400 +('+string+' -1)/12) AS timestamp_year,loan_id from josh_mody_ntemp ;')
     con.execute('CREATE TABLE finaltbl AS SELECT josh_mody_ntemp.timestamp_month AS timestamp_month,josh_mody_ntemp.delinquency_12 AS delinquency_12,josh_mody_ntemp.upb_12 AS upb_12,josh_mody_ntemp.josh_months AS josh_months,josh_mody_ntemp.loan_id AS loan_id ,josh_mody_ntemp.josh_mody_n AS josh_mody_n,josh_mody_ntemp.delinquency_12 AS delinquency_12,timestamp_yeartemp.timestamp_year AS timestamp_year FROM josh_mody_ntemp  LEFT JOIN timestamp_yeartemp  ON timestamp_yeartemp.loan_id= josh_mody_ntemp.loan_id;');
     con.execute('DROP TABLE IF EXISTS tmpdf;')
     con.execute('DROP TABLE IF EXISTS josh_monthstemp;')
     con.execute('DROP TABLE IF EXISTS deling_12 ;');
     con.execute('DROP TABLE IF EXISTS josh_monthsjoin ;');
     con.execute('DROP TABLE IF EXISTS josh_mody_ntemp ;');
     con.execute('DROP TABLE IF EXISTS testdfs;')
    # con.execute('DROP TABLE IF EXISTS joined_df ;');
     con.execute('ALTER TABLE finaltbl RENAME TO testdfs;')
     #tmpdf['timestamp_month'] = np.int8(y)
    # tmpdf.drop(columns=['josh_mody_n'], inplace=True)
     #testdfs.append(tmpdf)
    # del(tmpdf)
    #del(joined_df)
   #return pd.concat(testdfs)


def combine_joined_12_mon(joined_df, testdf, **kwargs):
    #joined_df.drop(columns=['delinquency_12', 'upb_12'], inplace=True)
    #joined_df['timestamp_year'] = joined_df['timestamp_year'].astype('int16')
    #joined_df['timestamp_month'] = joined_df['timestamp_month'].astype('int8')
    con.execute('DROP TABLE IF EXISTS join_final ;')
    con.execute('CREATE TABLE join_final AS SELECT testdfs.timestamp_year AS timestamp_year ,testdfs.timestamp_month AS timestamp_month,testdfs.loan_id AS loan_id  FROM testdfs  LEFT JOIN joined_df  ON testdfs.loan_id= joined_df.loan_id;');
    con.execute('DROP TABLE IF EXISTS joined_df ;')
    con.execute('ALTER TABLE join_final RENAME TO joined_df;')
    con.execute('DROP TABLE IF EXISTS join_final ;')

    #return joined_df.merge(testdf, how='left', on=['loan_id', 'timestamp_year', 'timestamp_month'])


def final_performance_delinquency(pdf, joined_df, **kwargs):
     merged = pdf[['loan_id', 'monthly_reporting_period']]
 #everdf1 = pdf[['loan_id', 'current_loan_delinquency_status']]
     con.execute('DROP TABLE IF EXISTS mergedtemp;')
     con.create_table('mergedtemp', merged)
    #merged['timestamp_month'] = merged['monthly_reporting_period'].dt.month
    #merged['timestamp_month'] = merged['timestamp_month'].astype('int8')
    #merged['timestamp_year'] = merged['monthly_reporting_period'].dt.year
    #merged['timestamp_year'] = merged['timestamp_year'].astype('int16')
    #merged = merged.merge(joined_df, how='left', on=['loan_id', 'timestamp_year', 'timestamp_month'])
     con.execute('DROP TABLE IF EXISTS merged_temp')
     con.execute( 'DROP TABLE IF EXISTS merged')
    # con.execute('SELECT loan_id,monthly_reporting_period FROM merged;');
     con.execute('CREATE TABLE merged_temp AS SELECT EXTRACT (MONTH FROM monthly_reporting_period) timestamp_month,EXTRACT(YEAR FROM monthly_reporting_period) timestamp_year,loan_id FROM mergedtemp ;');
     con.execute('CREATE TABLE merged AS SELECT joined_df.timestamp_year AS time_stamp_year,joined_df.timestamp_month AS time_stamp_month,joined_df.loan_id AS loan_id  FROM merged_temp LEFT JOIN joined_df ON merged_temp.loan_id=joined_df.loan_id AND merged_temp.timestamp_year=joined_df.timestamp_year AND merged_temp.timestamp_month=joined_df.timestamp_month');
     #con.execute('UPDATE merged SET time_stamp_year = CAST (time_stamp_year  AS DATE ENCODING FIXED(16));');
     con.execute('DROP TABLE IF EXISTS merged_temp')
     #merged.drop(columns=['timestamp_year'], inplace=True)
     #merged.drop(columns=['timestamp_month'], inplace=True)
     return merged
def join_perf_acq_pdfs(perf, acq, **kwargs):
     #return perf.merge(acq, how='left', on=['loan_id'])
    con.execute('DROP TABLE IF EXISTS tempperf ');
    con.execute('CREATE TABLE tempperf AS SELECT acq.loan_id AS loan_id,acq.seller_name FROM acq LEFT JOIN perf ON acq.loan_id = perf.loan_id;');


def last_mile_cleaning(df, **kwargs):
    #for col, dtype in df.dtypes.iteritems():
    #    if str(dtype)=='category':
    #        df[col] = df[col].cat.codes
    #df['delinquency_12'] = df['delinquency_12'] > 0
    #df['delinquency_12'] = df['delinquency_12'].fillna(False).astype('int32')
     return df #.to_arrow(index=False)


year = 2000
quarter = 1
perf_file = os.path.join(data_directory, "perf", "Performance_" + str(year) + "Q" + str(quarter) + ".txt")
#pdf = run_pd_workflow(year=year, quarter=quarter, perf_file=perf_file)
#print(pdf)
t1 = time.time()
pdf = run_pd_workflow(year=year, quarter=quarter, perf_file=perf_file)
t2 = time.time()
print("Total exec time:", t2-t1)


# start_year = 2000
# end_year = 2017

# pd_dfs = []
# pd_time = 0
# quarter = 1
# year = start_year
# while year != end_year:
#     for file in glob(os.path.join(data_directory, "Performance_" + str(year) + "Q" + str(quarter) + "*")):
#         pd_dfs.append(process_quarter_pd(year=year, quarter=quarter, perf_file=file))
#     quarter += 1
#     if quarter == 5:
#         year += 1
#         quarter = 1
# wait(pd_dfs)
(base) avanipat@ansatlin12:/localdisk2/avanipat/omniscidb/build$
