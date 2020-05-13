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


def groupby_query1_modin(x, queries_results):
    query_name = "groupby_query1"
    question = "sum v1 by id1"  # 1
    gc.collect()
    t_start = timer()
    ans = x.groupby(["id1"]).agg({"v1": "sum"})
    print(ans.shape)
    queries_results[query_name]["t_run1"] = timer() - t_start
    m = memory_usage()
    t_start = timer()
    chk = [ans["v1"].sum()]
    queries_results[query_name]["chk_t_run1"] = timer() - t_start
    chk = make_chk(chk)
    print(
        query_name,
        ", question:",
        question,
        ",run1",
        ",in_rows:",
        x.shape[0],
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
    ans = x.groupby(["id1"]).agg({"v1": "sum"})
    print(ans.shape)
    queries_results[query_name]["t_run2"] = timer() - t_start
    m = memory_usage()
    t_start = timer()
    chk = [ans["v1"].sum()]
    queries_results[query_name]["chk_t_run2"] = timer() - t_start
    chk = make_chk(chk)
    print(
        query_name,
        ", question:",
        question,
        ",run2",
        ",in_rows:",
        x.shape[0],
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


def groupby_query2_modin(x, queries_results):
    query_name = "groupby_query2"
    question = "sum v1 by id1:id2"  # 2
    gc.collect()
    t_start = timer()
    ans = x.groupby(["id1", "id2"]).agg({"v1": "sum"})
    print(ans.shape)
    queries_results[query_name]["t_run1"] = timer() - t_start
    m = memory_usage()
    t_start = timer()
    chk = [ans["v1"].sum()]
    queries_results[query_name]["chk_t_run1"] = timer() - t_start
    print(
        query_name,
        ", question:",
        question,
        ",run1",
        ",in_rows:",
        x.shape[0],
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
    ans = x.groupby(["id1", "id2"]).agg({"v1": "sum"})
    print(ans.shape)
    queries_results[query_name]["t_run2"] = timer() - t_start
    m = memory_usage()
    t_start = timer()
    chk = [ans["v1"].sum()]
    queries_results[query_name]["chk_t_run2"] = timer() - t_start
    print(
        query_name,
        " question:",
        question,
        ",run2",
        ",in_rows:",
        x.shape[0],
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


def groupby_query3_modin(x, queries_results):
    query_name = "groupby_query3"
    question = "sum v1 mean v3 by id3"  # 3
    gc.collect()
    t_start = timer()
    ans = x.groupby(["id3"]).agg({"v1": "sum", "v3": "mean"})
    print(ans.shape)
    queries_results[query_name]["t_run1"] = timer() - t_start
    m = memory_usage()
    t_start = timer()
    chk = [ans["v1"].sum(), ans["v3"].sum()]
    queries_results[query_name]["chk_t_run1"] = timer() - t_start
    print(
        query_name,
        ", question:",
        question,
        ",run1",
        ",in_rows:",
        x.shape[0],
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
    ans = x.groupby(["id3"]).agg({"v1": "sum", "v3": "mean"})
    print(ans.shape)
    queries_results[query_name]["t_run2"] = timer() - t_start
    m = memory_usage()
    t_start = timer()
    chk = [ans["v1"].sum(), ans["v3"].sum()]
    queries_results[query_name]["chk_t_run2"] = timer() - t_start
    print(
        "query3, question:",
        question,
        ",run2",
        ",in_rows:",
        x.shape[0],
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


def groupby_query4_modin(x, queries_results):
    query_name = "groupby_query4"
    question = "mean v1:v3 by id4"  # 4
    gc.collect()
    t_start = timer()
    ans = x.groupby(["id4"]).agg({"v1": "mean", "v2": "mean", "v3": "mean"})
    print(ans.shape)
    queries_results[query_name]["t_run1"] = timer() - t_start
    m = memory_usage()
    t_start = timer()
    chk = [ans["v1"].sum(), ans["v2"].sum(), ans["v3"].sum()]
    queries_results[query_name]["chk_t_run1"] = timer() - t_start
    chk = make_chk(chk)
    print(
        query_name,
        ", question:",
        question,
        ",run1",
        ",in_rows:",
        x.shape[0],
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
    ans = x.groupby(["id4"]).agg({"v1": "mean", "v2": "mean", "v3": "mean"})
    print(ans.shape)
    queries_results[query_name]["t_run2"] = timer() - t_start
    m = memory_usage()
    t_start = timer()
    chk = [ans["v1"].sum(), ans["v2"].sum(), ans["v3"].sum()]
    queries_results[query_name]["chk_t_run2"] = timer() - t_start
    chk = make_chk(chk)
    print(
        query_name,
        ", question:",
        question,
        ",run2",
        ",in_rows:",
        x.shape[0],
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


def groupby_query5_modin(x, queries_results):
    query_name = "groupby_query5"
    question = "sum v1:v3 by id6"  # 5
    gc.collect()
    t_start = timer()
    ans = x.groupby(["id6"]).agg({"v1": "sum", "v2": "sum", "v3": "sum"})
    print(ans.shape)
    queries_results[query_name]["t_run1"] = timer() - t_start
    m = memory_usage()
    t_start = timer()
    chk = [ans["v1"].sum(), ans["v2"].sum(), ans["v3"].sum()]
    queries_results[query_name]["chk_t_run1"] = timer() - t_start
    chk = make_chk(chk)
    print(
        query_name,
        ", question:",
        question,
        ",run1",
        ",in_rows:",
        x.shape[0],
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
    ans = x.groupby(["id6"]).agg({"v1": "sum", "v2": "sum", "v3": "sum"})
    print(ans.shape)
    queries_results[query_name]["t_run2"] = timer() - t_start
    m = memory_usage()
    t_start = timer()
    chk = [ans["v1"].sum(), ans["v2"].sum(), ans["v3"].sum()]
    queries_results[query_name]["chk_t_run2"] = timer() - t_start
    chk = make_chk(chk)
    print(
        query_name,
        ", question:",
        question,
        ",run2",
        ",in_rows:",
        x.shape[0],
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


def queries_modin(filename, pandas_mode):
    queries = {
        "groupby_query1": groupby_query1_modin,
        "groupby_query2": groupby_query2_modin,
        "groupby_query3": groupby_query3_modin,
        "groupby_query4": groupby_query4_modin,
        "groupby_query5": groupby_query5_modin,
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
