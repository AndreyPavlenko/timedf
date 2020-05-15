# coding: utf-8
import os
import sys
import time
import traceback
import warnings
from timeit import default_timer as timer
import gc

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils import (
    check_fragments_size,
    cod,
    compare_dataframes,
    import_pandas_into_module_namespace,
    load_data_pandas,
    mse,
    print_results,
    make_chk,
    memory_usage,
)

warnings.filterwarnings("ignore")


def execute_groupby_query_chk_expr_v1(ans): # q1, q2
    return [ans["v1"].sum()]


def execute_groupby_query_chk_expr_v2(ans): # q3
    return [ans["v1"].sum(), ans["v3"].sum()]


def execute_groupby_query_chk_expr_v3(ans): # q4, q5
    return [ans["v1"].sum(), ans["v2"].sum(), ans["v3"].sum()]


def execute_groupby_query_chk_expr_v4(ans): # q6
    return [ans['v3']['median'].sum(), ans['v3']['std'].sum()]


def execute_groupby_query_chk_expr_v5(ans): # q7
    return [ans['range_v1_v2'].sum()]


def execute_groupby_query_chk_expr_v6(ans): # q8
    return [ans['v3'].sum()]


groupby_queries_chk_funcs = {
    "groupby_query1": execute_groupby_query_chk_expr_v1,
    "groupby_query2": execute_groupby_query_chk_expr_v1,
    "groupby_query3": execute_groupby_query_chk_expr_v2,
    "groupby_query4": execute_groupby_query_chk_expr_v3,
    "groupby_query5": execute_groupby_query_chk_expr_v3,
    "groupby_query6": execute_groupby_query_chk_expr_v4,
    "groupby_query7": execute_groupby_query_chk_expr_v5,
    "groupby_query8": execute_groupby_query_chk_expr_v6,
}

def execute_groupby_query_expr_v1(x, groupby_cols, agg_cols_funcs): # q1, q2, q3, q4, q5, q6
    return x.groupby(groupby_cols).agg(agg_cols_funcs)


def execute_groupby_query_expr_v2(x, groupby_cols, agg_cols_funcs, range_cols): # q7
    return x.groupby(groupby_cols).agg(agg_cols_funcs).assign(range_v1_v2=lambda x: x[range_cols[0]] - x[range_cols[0]])[['range_v1_v2']]

def execute_groupby_query_expr_v3(x, select_cols, sort_col, sort_ascending, groupby_cols): # q8
    return x[select_cols].sort_values(sort_col, ascending=sort_ascending).groupby(groupby_cols).head(2)


groupby_queries_funcs = {
    "groupby_query1": execute_groupby_query_expr_v1,
    "groupby_query2": execute_groupby_query_expr_v1,
    "groupby_query3": execute_groupby_query_expr_v1,
    "groupby_query4": execute_groupby_query_expr_v1,
    "groupby_query5": execute_groupby_query_expr_v1,
    "groupby_query6": execute_groupby_query_expr_v1,
    "groupby_query7": execute_groupby_query_expr_v2,
    "groupby_query8": execute_groupby_query_expr_v3,
}


def execute_groupby_query(query_args, queries_results, query_name, question):
    gc.collect()
    t_start = timer()
    ans = groupby_queries_funcs[query_name](**query_args)
    print(ans.shape)
    queries_results[query_name]["t_run1"] = timer() - t_start
    m = memory_usage()
    t_start = timer()
    chk = groupby_queries_chk_funcs[query_name](ans)
    queries_results[query_name]["chk_t_run1"] = timer() - t_start
    chk = make_chk(chk)
    print(
        query_name,
        ", question:",
        question,
        ",run1",
        ",in_rows:",
        query_args["x"].shape[0],
        ",out_rows:",
        ans.shape[0],
        ",out_cols:",
        ans.shape[1],
        ",time_sec:",
        queries_results[query_name]["t_run1"],
        "mem_gb:",
        m,
        ",chk:",
        chk,
        ",chk_time_sec:",
        queries_results[query_name]["chk_t_run1"],
    )
    del ans

    gc.collect()
    t_start = timer()
    ans = groupby_queries_funcs[query_name](**query_args)
    print(ans.shape)
    queries_results[query_name]["t_run2"] = timer() - t_start
    m = memory_usage()
    t_start = timer()
    chk = groupby_queries_chk_funcs[query_name](ans)
    queries_results[query_name]["chk_t_run2"] = timer() - t_start
    chk = make_chk(chk)
    print(
        query_name,
        ", question:",
        question,
        ",run2",
        ",in_rows:",
        query_args["x"].shape[0],
        ",out_rows:",
        ans.shape[0],
        ",out_cols:",
        ans.shape[1],
        ",time_sec:",
        queries_results[query_name]["t_run2"],
        "mem_gb:",
        m,
        ",chk:",
        chk,
        ",chk_time_sec:",
        queries_results[query_name]["chk_t_run2"],
    )
    del ans


def groupby_query1_modin(x, queries_results):
    query_name = "groupby_query1"
    question = "sum v1 by id1"  # 1
    query_args = {"x": x, "groupby_cols": ["id1"], "agg_cols_funcs": {"v1": "sum"}}

    execute_groupby_query(
        query_args=query_args,
        queries_results=queries_results,
        query_name=query_name,
        question=question,
    )


def groupby_query2_modin(x, queries_results):
    query_name = "groupby_query2"
    question = "sum v1 by id1:id2"  # 2
    query_args = {"x": x, "groupby_cols": ["id1", "id2"], "agg_cols_funcs": {"v1": "sum"}}

    execute_groupby_query(
        query_args=query_args,
        queries_results=queries_results,
        query_name=query_name,
        question=question,
    )


def groupby_query3_modin(x, queries_results):
    query_name = "groupby_query3"
    question = "sum v1 mean v3 by id3"  # 3
    query_args = {"x": x, "groupby_cols": ["id3"], "agg_cols_funcs": {"v1": "sum", "v3": "mean"}}

    execute_groupby_query(
        query_args=query_args,
        queries_results=queries_results,
        query_name=query_name,
        question=question,
    )


def groupby_query4_modin(x, queries_results):
    query_name = "groupby_query4"
    question = "mean v1:v3 by id4"  # 4
    query_args = {"x": x, "groupby_cols": ["id4"], "agg_cols_funcs": {"v1": "mean", "v2": "mean", "v3": "mean"}}

    execute_groupby_query(
        query_args=query_args,
        queries_results=queries_results,
        query_name=query_name,
        question=question,
    )


def groupby_query5_modin(x, queries_results):
    query_name = "groupby_query5"
    question = "sum v1:v3 by id6"  # 5
    query_args = {"x": x, "groupby_cols": ["id6"], "agg_cols_funcs": {"v1": "sum", "v2": "sum", "v3": "sum"}}

    execute_groupby_query(
        query_args=query_args,
        queries_results=queries_results,
        query_name=query_name,
        question=question,
    )

def groupby_query6_modin(x, queries_results):
    query_name = "groupby_query6"
    question = "median v3 sd v3 by id4 id5" # q6
    query_args = {"x": x, "groupby_cols": ['id4','id5'], "agg_cols_funcs": {'v3': ['median','std']}}

    execute_groupby_query(
        query_args=query_args,
        queries_results=queries_results,
        query_name=query_name,
        question=question,
    )

def groupby_query7_modin(x, queries_results):
    query_name = "groupby_query7"
    question = "max v1 - min v2 by id3" # q7
    query_args = {"x": x, "groupby_cols": ['id3'], "agg_cols_funcs": {'v1': 'max', 'v2': 'min'}, "range_cols": ["v1", "v2"]}

    execute_groupby_query(
        query_args=query_args,
        queries_results=queries_results,
        query_name=query_name,
        question=question,
    )


def groupby_query8_modin(x, queries_results):
    query_name = "groupby_query8"
    question = "largest two v3 by id6" # q8
    query_args = {"x": x, "select_cols": ['id6','v3'], "sort_col": "v3", "sort_ascending": False, "groupby_cols": ['id6']}

    execute_groupby_query(
        query_args=query_args,
        queries_results=queries_results,
        query_name=query_name,
        question=question,
    )


#     execute_groupby_query(
#         x=x,
#         queries_results=queries_results,
#         query_name=query_name,
#         question=question,
#         groupby_cols=['id4','id5'],
#         agg_cols_funcs={'v3': ['median','std']},
#     )

# def groupby_query6_modin(x, queries_results):
#     query_name = "groupby_query6"
#     question = "median v3 sd v3 by id4 id5" # q6

#     execute_groupby_query(
#         x=x,
#         queries_results=queries_results,
#         query_name=query_name,
#         question=question,
#         groupby_cols=['id4','id5'],
#         agg_cols_funcs={'v3': ['median','std']},
#     )

# def join_query1_modin(x, y, queries_results):
#     query_name = "groupby_query1"
#     question = "sum v1 by id1"  # 1

#     execute_groupby_query(
#         x=x,
#         queries_results=queries_results,
#         query_name=query_name,
#         question=question,
#         groupby_cols=["id1"],
#         agg_cols_funcs={"v1": "sum"},
#     )

def queries_modin(filename, pandas_mode):
    queries = {
        "groupby_query1": groupby_query1_modin,
        "groupby_query2": groupby_query2_modin,
        "groupby_query3": groupby_query3_modin,
        "groupby_query4": groupby_query4_modin,
        "groupby_query5": groupby_query5_modin,
        "groupby_query6": groupby_query6_modin,
        "groupby_query7": groupby_query7_modin,
        "groupby_query8": groupby_query8_modin,
        # "join_query1": join_query1_modin,
    }
    groupby_queries_results_fields = ["t_run1", "chk_t_run1", "t_run2", "chk_t_run2"]
    queries_results = {x + "_run1_t": 0.0 for x in queries.keys()}
    queries_results = {x: {y: 0.0 for y in groupby_queries_results_fields} for x in queries.keys()}
    data_file_size = os.path.getsize(filename) / 1024 / 1024

    print(f"loading dataset {filename}")
    t0 = timer()
    x = pd.read_csv(filename)
    data_file_import_time = timer() - t0

    queries_parameters = {"x": x, "queries_results": queries_results}
    for query_name, query_func in queries.items():
        query_func(**queries_parameters)
        print(f"{pandas_mode} {query_name} results:")
        print_results(results=queries_results[query_name], unit="s")
        queries_results[query_name]["Backend"] = pandas_mode
        queries_results[query_name]["t_readcsv"] = data_file_import_time
        queries_results[query_name]["dataset_size"] = data_file_size

    return queries_results


def run_benchmark(parameters):
    ignored_parameters = {
        "dfiles_num": parameters["dfiles_num"],
        "gpu_memory": parameters["gpu_memory"],
        "no_ml": parameters["no_ml"],
        "no_ibis": parameters["no_ibis"],
        "optimizer": parameters["optimizer"],
        "validation": parameters["validation"],
    }
    warnings.warn(f"Parameters {ignored_parameters} are irnored", RuntimeWarning)

    parameters["data_file"] = parameters["data_file"].replace("'", "")

    queries_times_modin = None

    try:
        if not parameters["no_pandas"]:
            import_pandas_into_module_namespace(
                namespace=run_benchmark.__globals__,
                mode=parameters["pandas_mode"],
                ray_tmpdir=parameters["ray_tmpdir"],
                ray_memory=parameters["ray_memory"],
            )
            queries_results = queries_modin(
                filename=parameters["data_file"], pandas_mode=parameters["pandas_mode"]
            )

        return {"ETL": [queries_results], "ML": []}
    except Exception:
        traceback.print_exc(file=sys.stdout)
        sys.exit(1)
