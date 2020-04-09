import warnings

import ibis

from utils import (
    cod,
    compare_dataframes,
    import_pandas_into_module_namespace,
    load_data_pandas,
    mse,
    print_results,
    split,
)
from .mortgage_ibis import etl_ibis

warnings.filterwarnings("ignore")


# Dataset link
# https://rapidsai.github.io/demos/datasets/mortgage-data


def run_benchmark(parameters):
    '''
        parameters = {
            "data_file": args.data_file,
            "dfiles_num": args.dfiles_num,
            "no_ml": args.no_ml,
            "no_ibis": args.no_ibis,
            "optimizer": args.optimizer,
            "pandas_mode": args.pandas_mode,
            "ray_tmpdir": args.ray_tmpdir,
            "ray_memory": args.ray_memory,
            "gpu_memory": args.gpu_memory,
            "validation": False if args.no_ibis else args.validation,
        }
        parameters["database_name"] = args.database_name
        parameters["table"] = args.table
        parameters["dnd"] = args.dnd
        parameters["dni"] = args.dni
        parameters["import_mode"] = args.import_mode
    '''
    parameters["data_file"] = parameters["data_file"].replace("'", "")
    ignored_parameters = {
        "gpu_memory": parameters["gpu_memory"],
    }
    warnings.warn(f"Parameters {ignored_parameters} are irnored", RuntimeWarning)
    if parameters['validation']:
        print('WARNING: Validation not yet supported')
    if parameters['import_mode'] not in ('fsi', 'pandas'):
        raise ValueError('Unsupported import mode: %s' % parameters['import_mode'])
    if parameters['dfiles_num'] != 1:
        raise NotImplementedError('Loading more than 1 file not implemented yet')
    

    import_pandas_into_module_namespace(
        namespace=run_benchmark.__globals__,
        mode=parameters["pandas_mode"],
        ray_tmpdir=parameters["ray_tmpdir"],
        ray_memory=parameters["ray_memory"],
    )

    acq_schema = ibis.Schema(
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
    perf_schema = ibis.Schema(
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
    etl_keys = ["t_readcsv", "t_etl"]


    result = {'ETL': [], 'ML': []}

    if not parameters['no_ibis']:
        df_ibis, x_ibis, y_ibis, etl_times_ibis = etl_ibis(
            dataset_path=parameters['data_file'],
            dfiles_num=parameters['dfiles_num'],
            acq_schema=acq_schema,
            perf_schema=perf_schema,
            database_name=parameters['database_name'],
            table_prefix=parameters['table'],
            omnisci_server_worker=parameters['omnisci_server_worker'],
            delete_old_database=not parameters['dnd'],
            create_new_table=not parameters['dni'],
            ipc_connection=parameters['ipc_connection'],
            etl_keys=etl_keys,
            import_mode=parameters['import_mode']
        )
        print_results(results=etl_times_ibis, backend='Ibis', unit='ms')
        etl_times_ibis['Backend'] = 'Ibis'
        result['ETL'].append(etl_times_ibis)

    return result
