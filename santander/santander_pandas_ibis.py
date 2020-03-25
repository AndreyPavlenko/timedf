# coding: utf-8

import os
import sys
import time
import traceback
import warnings
from timeit import default_timer as timer

import ibis

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils import (
    cod,
    # compare_dataframes,
    import_pandas_into_module_namespace,
    load_data_pandas,
    mse,
    print_times,
)

warnings.filterwarnings("ignore")


def compare_tables(table1, table2):
    if table1.equals(table2):
        print("Tables are equal")
        return True
    else:
        print("\ntables are not equal, table1:")
        print(table1.info())
        print(table1)
        print("\ntable2:")
        print(table2.info())
        print(table2)
        return False


def compare_dataframes(ibis_df, pandas_df):
    comparison_result = True
    for i in range(len(ibis_df)):
        ibis_df[i].index = pandas_df[i].index
        comparison_result = comparison_result and ibis_df[i].equals(pandas_df[i])

    if comparison_result:
        print("Tables are equal")
        return True
    else:
        diff = {}
        for i in range(len(ibis_df)):
            diff_df = ibis_df[i] - pandas_df[i]
            if len(diff_df.shape) > 1:
                diff["DataFrame %s max deviation" % str(i + 1)] = diff_df.max().max()
                diff["DataFrame %s min deviation" % str(i + 1)] = diff_df.min().min()
            else:
                diff["DataFrame %s max deviation" % str(i + 1)] = diff_df.max()
                diff["DataFrame %s min deviation" % str(i + 1)] = diff_df.min()

        check_res = True
        lowest_order = 0.0001
        print("Values check summary:")
        for dev_type, value in diff.items():
            print(dev_type, ":", value)
            check_res = check_res and (abs(value) < lowest_order)

        if check_res == True:
            print(
                "Deviation of values is lower, than values smallest order -> tables are equal"
            )
        else:
            print(
                "Deviation of values is higher, than values smallest order -> tables are unequal"
            )
        return check_res


# Dataset link
# https://www.kaggle.com/c/santander-customer-transaction-prediction/data

# Current script prerequisites:
# 1) Patched OmniSci version (https://github.com/intel-go/omniscidb/tree/ienkovich/santander)
# 2) Ibis version not older than e60d1af commit (otherwise apply ibis-santander.patch patch)

def etl_pandas(filename, columns_names, columns_types):
    etl_times = {
        "t_groupby_agg": 0.0,
        "t_drop": 0.0,
        "t_merge": 0.0,
        "t_readcsv": 0.0,
        "t_train_test_split": 0.0,
        "t_where": 0.0,
        "t_reset_index": 0.0,
        "t_assign_data": 0.0,
        "t_etl": 0.0,
    }

    t0 = timer()
    train_pd = load_data_pandas(
        filename=filename,
        columns_names=columns_names,
        columns_types=columns_types,
        header=0,
        nrows=None,
        use_gzip=filename.endswith(".gz"),
        pd=run_benchmark.__globals__["pd"],
    )
    etl_times["t_readcsv"] = timer() - t0

    t_etl_begin = timer()

    for i in range(200):
        col = "var_%d" % i
        t0 = timer()
        var_count = train_pd.groupby(col).agg({col: "count"})
        etl_times["t_groupby_agg"] += timer() - t0

        t0 = timer()
        var_count.columns = ["%s_count" % col]
        var_count = var_count.reset_index()
        etl_times["t_reset_index"] += timer() - t0

        t0 = timer()
        train_pd = train_pd.merge(var_count, on=col, how="left")
        etl_times["t_merge"] += timer() - t0

    for i in range(200):
        col = "var_%d" % i

        t0 = timer()
        mask = train_pd["%s_count" % col] > 1
        etl_times["t_where"] += timer() - t0

        t0 = timer()
        train_pd.loc[mask, "%s_gt1" % col] = train_pd.loc[mask, col]
        etl_times["t_assign_data"] += timer() - t0

    # train, test data split
    t0 = timer()
    train, valid = train_pd[:-10000], train_pd[-10000:]
    etl_times["t_train_test_split"] = timer() - t0

    t0 = timer()
    x_train = train.drop(["target", "ID_code"], axis=1)
    etl_times["t_drop"] += timer() - t0

    y_train = train["target"]

    t0 = timer()
    x_test = valid.drop(["target", "ID_code"], axis=1)
    etl_times["t_drop"] += timer() - t0

    y_test = valid["target"]

    etl_times["t_etl"] = timer() - t_etl_begin

    return x_train, y_train, x_test, y_test, etl_times


def etl_ibis(
        filename,
        columns_names,
        columns_types,
        database_name,
        table_name,
        omnisci_server_worker,
        delete_old_database,
        create_new_table,
        connection_func,
        validation,
        run_import_queries,
):
    tmp_table_name = "tmp_table"

    etl_times = {"t_groupby_merge_where": 0.0, "t_train_test_split": 0.0, "t_etl": 0.0}

    omnisci_server_worker.create_database(
        database_name, delete_if_exists=delete_old_database
    )

    time.sleep(2)
    omnisci_server_worker.connect_to_server(database=database_name)

    if run_import_queries:
        etl_times_import = {
            "t_readcsv_by_ibis": 0.0,
            "t_readcsv_by_COPY": 0.0,
            "t_readcsv_by_FSI": 0.0,
        }

        # SQL statemnts preparation for data file import queries
        connect_to_db_sql_template = "\c {0} admin HyperInteractive"
        create_table_sql_template = """
        CREATE TABLE {0} ({1});
        """
        import_by_COPY_sql_template = """
        COPY {0} FROM '{1}' WITH (header='{2}');
        """
        import_by_FSI_sql_template = """
        CREATE TEMPORARY TABLE {0} ({1}) WITH (storage_type='CSV:{2}');
        """
        drop_table_sql_template = """
        DROP TABLE IF EXISTS {0};
        """

        import_query_cols_list = (
                ["ID_code TEXT ENCODING NONE, \n", "target SMALLINT, \n"]
                + ["var_%s DOUBLE, \n" % i for i in range(199)]
                + ["var_199 DOUBLE"]
        )
        import_query_cols_str = "".join(import_query_cols_list)

        create_table_sql = create_table_sql_template.format(
            tmp_table_name, import_query_cols_str
        )
        import_by_COPY_sql = import_by_COPY_sql_template.format(
            tmp_table_name, filename, "true"
        )
        import_by_FSI_sql = import_by_FSI_sql_template.format(
            tmp_table_name, import_query_cols_str, filename
        )

        # data file import by ibis
        columns_types_import_query = ["string", "int64"] + [
            "float64" for _ in range(200)
        ]
        schema_table_import = ibis.Schema(
            names=columns_names, types=columns_types_import_query
        )
        omnisci_server_worker.get_conn().create_table(
            table_name=tmp_table_name,
            schema=schema_table_import,
            database=database_name,
        )

        table_import_query = omnisci_server_worker.database(database_name).table(tmp_table_name)
        t0 = timer()
        table_import_query.read_csv(filename, delimiter=",")
        etl_times_import["t_readcsv_by_ibis"] = timer() - t0

        # data file import by FSI
        omnisci_server_worker.drop_table(tmp_table_name)
        t0 = timer()
        omnisci_server_worker.execute_sql_query(import_by_FSI_sql)
        etl_times_import["t_readcsv_by_FSI"] = timer() - t0

        omnisci_server_worker.drop_table(tmp_table_name)

        # data file import by SQL COPY statement
        omnisci_server_worker.execute_sql_query(create_table_sql)

        t0 = timer()
        omnisci_server_worker.execute_sql_query(import_by_COPY_sql)
        etl_times_import["t_readcsv_by_COPY"] = timer() - t0
        print_times(times=etl_times_import)

        omnisci_server_worker.drop_table(tmp_table_name)

    if create_new_table:
        # Create table and import data for ETL queries
        schema_table = ibis.Schema(names=columns_names, types=columns_types)
        omnisci_server_worker.get_conn().create_table(
            table_name=table_name,
            schema=schema_table,
            database=database_name,
        )

        table_import = omnisci_server_worker.database(database_name).table(table_name)
        table_import.read_csv(filename, delimiter=",")

    conn = connection_func()
    db = conn.database(database_name)
    table = db.table(table_name)

    # group_by/count, merge (join) and filtration queries
    # We are making 400 columns and then insert them into original table thus avoiding
    # nested sql requests
    t0 = timer()
    count_cols = []
    orig_cols = ["ID_code", "target"] + ['var_%s' % i for i in range(200)]
    cast_cols = []
    cast_cols.append(table["target"].cast("int64").name("target0"))
    gt1_cols = []
    for i in range(200):
        col = "var_%d" % i
        col_count = "var_%d_count" % i
        col_gt1 = "var_%d_gt1" % i
        w = ibis.window(group_by=col)
        count_cols.append(table[col].count().over(w).name(col_count))
        gt1_cols.append(
            ibis.case()
                .when(
                table[col].count().over(w).name(col_count) > 1,
                table[col].cast("float32"),
            )
                .else_(ibis.null())
                .end()
                .name("var_%d_gt1" % i)
        )
        cast_cols.append(table[col].cast("float32").name(col))

    table = table.mutate(count_cols)
    table = table.drop(orig_cols)
    table = table.mutate(gt1_cols)
    table = table.mutate(cast_cols)

    table_df = table.execute()
    etl_times["t_groupby_merge_where"] = timer() - t0

    # rows split query
    t0 = timer()
    training_part, validation_part = table_df[:-10000], table_df[-10000:]
    etl_times["t_train_test_split"] = timer() - t0

    etl_times["t_etl"] = etl_times["t_groupby_merge_where"] + etl_times["t_train_test_split"]

    x_train = training_part.drop(['target0'], axis=1)
    y_train = training_part['target0']
    x_test = validation_part.drop(['target0'], axis=1)
    y_test = validation_part['target0']

    return x_train, y_train, x_test, y_test, etl_times


def ml(x_train, y_train, x_test, y_test):
    import xgboost

    ml_times = {"t_ML": 0.0, "t_train": 0.0, "t_inference": 0.0, "t_dmatrix": 0.0}
    ml_scores = {"mse": 0.0, "cod": 0.0}

    t0 = timer()
    training_dmat_part = xgboost.DMatrix(data=x_train, label=y_train)
    testing_dmat_part = xgboost.DMatrix(data=x_test, label=y_test)
    ml_times["t_dmatrix"] = timer() - t0

    watchlist = [(training_dmat_part, "eval"), (testing_dmat_part, "train")]
    xgb_params = {
        "objective": "binary:logistic",
        "tree_method": "hist",
        "max_depth": 1,
        "nthread": 56,
        "eta": 0.1,
        "silent": 1,
        "subsample": 0.5,
        "colsample_bytree": 0.05,
        "eval_metric": "auc",
    }

    t0 = timer()
    model = xgboost.train(
        xgb_params,
        dtrain=training_dmat_part,
        num_boost_round=10000,
        evals=watchlist,
        early_stopping_rounds=30,
        maximize=True,
        verbose_eval=1000,
    )
    ml_times["t_train"] = timer() - t0

    t0 = timer()
    yp = model.predict(testing_dmat_part)
    ml_times["t_inference"] = timer() - t0

    ml_scores["mse"] = mse(y_test, yp)
    ml_scores["cod"] = cod(y_test, yp)

    ml_times["t_ML"] += ml_times["t_train"] + ml_times["t_inference"]

    return ml_scores, ml_times


def run_benchmark(parameters):
    ignored_parameters = {
        "dfiles_num": parameters["dfiles_num"],
        "gpu_memory": parameters["gpu_memory"],
    }
    warnings.warn(f"Parameters {ignored_parameters} are irnored", RuntimeWarning)

    parameters["data_file"] = parameters["data_file"].replace("'", "")

    var_cols = ["var_%s" % i for i in range(200)]
    count_cols = ["var_%s_count" % i for i in range(200)]
    gt1_cols = ["var_%s_gt1" % i for i in range(200)]
    columns_names = ["ID_code", "target"] + var_cols
    columns_types_pd = ["object", "int64"] + ["float64" for _ in range(200)]
    columns_types_ibis = ["string", "int32"] + ["decimal(8, 4)" for _ in range(200)]

    try:

        import_pandas_into_module_namespace(
            namespace=run_benchmark.__globals__,
            mode=parameters["pandas_mode"],
            ray_tmpdir=parameters["ray_tmpdir"],
            ray_memory=parameters["ray_memory"],
        )

        if not parameters["no_ibis"]:
            x_train_ibis, y_train_ibis, x_test_ibis, y_test_ibis, etl_times_ibis = etl_ibis(
                filename=parameters["data_file"],
                run_import_queries=False,
                columns_names=columns_names,
                columns_types=columns_types_ibis,
                database_name=parameters["database_name"],
                table_name=parameters["table"],
                omnisci_server_worker=parameters["omnisci_server_worker"],
                delete_old_database=not parameters["dnd"],
                create_new_table=not parameters["dni"],
                connection_func=parameters["connect_to_sever"],
                validation=parameters["validation"],
            )

            print_times(times=etl_times_ibis, backend="Ibis")
            etl_times_ibis["Backend"] = "Ibis"

        x_train, y_train, x_test, y_test, etl_times = etl_pandas(
            filename=parameters["data_file"],
            columns_names=columns_names,
            columns_types=columns_types_pd
        )
        print_times(times=etl_times, backend=parameters["pandas_mode"])
        etl_times["Backend"] = parameters["pandas_mode"]

        if not parameters["no_ml"]:
            ml_scores, ml_times = ml(
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
            )
            print_times(times=ml_times, backend=parameters["pandas_mode"])
            ml_times["Backend"] = parameters["pandas_mode"]
            print_times(times=ml_scores, backend=parameters["pandas_mode"])
            ml_scores["Backend"] = parameters["pandas_mode"]

            if not parameters["no_ibis"]:
                ml_scores_ibis, ml_times_ibis = ml(
                    x_train=x_train_ibis,
                    y_train=y_train_ibis,
                    x_test=x_test_ibis,
                    y_test=y_test_ibis,
                )
                print_times(times=ml_times_ibis, backend="Ibis")
                ml_times_ibis["Backend"] = "Ibis"
                print_times(times=ml_scores_ibis, backend="Ibis")
                ml_scores_ibis["Backend"] = "Ibis"

        # Results validation block (comparison of etl_ibis and etl_pandas outputs)
        if parameters["validation"] and not parameters["no_ibis"]:
            print("Validation of ETL query results with original input table ...")
            cols_to_sort = ['var_0', 'var_1', 'var_2', 'var_3', 'var_4']

            x_ibis = pd.concat([x_train_ibis, x_test_ibis])
            y_ibis = pd.concat([y_train_ibis, y_test_ibis])
            etl_ibis_res = pd.concat([x_ibis, y_ibis], axis=1)
            etl_ibis_res = etl_ibis_res.sort_values(by=cols_to_sort)
            x = pd.concat([x_train, x_test])
            y = pd.concat([y_train, y_test])
            etl_pandas_res = pd.concat([x, y], axis=1)
            etl_pandas_res = etl_pandas_res.sort_values(by=cols_to_sort)

            # print("Validating queries results (var_xx columns) ...")
            # compare_result1 = compare_dataframes(ibis_df=[etl_ibis_res[var_cols]],
            #                                      pandas_df=[etl_pandas_res[var_cols]])
            # print("Validating queries results (var_xx_count columns) ...")
            # compare_result2 = compare_dataframes(ibis_df=[etl_ibis_res[count_cols]],
            #                                      pandas_df=[etl_pandas_res[count_cols]])
            # print("Validating queries results (var_xx_gt1 columns) ...")
            # compare_result3 = compare_dataframes(ibis_df=[etl_ibis_res[gt1_cols]],
            #                                      pandas_df=[etl_pandas_res[gt1_cols]])
            # print("Validating queries results (target column) ...")
            # compare_result4 = compare_dataframes(ibis_df=[etl_ibis_res['target0']],
            #                                      pandas_df=[etl_pandas_res['target']])

            for score in ml_scores:
                if ml_scores[score] == ml_scores_ibis[score]:
                    print(f"{score} are equal")
                else:
                    print(
                        f"{score} aren't equal: Ibis={ml_scores_ibis[score]}, Pandas={ml_scores[score]}")

        return {"ETL": [etl_times_ibis, etl_times], "ML": [ml_times_ibis, ml_times]}
    except Exception:
        traceback.print_exc(file=sys.stdout)
        sys.exit(1)
