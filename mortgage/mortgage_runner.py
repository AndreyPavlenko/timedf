import warnings

from utils import (
    cod,
    compare_dataframes,
    import_pandas_into_module_namespace,
    load_data_pandas,
    mse,
    print_results,
    split,
)

warnings.filterwarnings("ignore")

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
    '''
    ignored_parameters = {
        "dfiles_num": parameters["dfiles_num"],
        "gpu_memory": parameters["gpu_memory"],
    }
    warnings.warn(f"Parameters {ignored_parameters} are irnored", RuntimeWarning)

    import_pandas_into_module_namespace(
        namespace=run_benchmark.__globals__,
        mode=parameters["pandas_mode"],
        ray_tmpdir=parameters["ray_tmpdir"],
        ray_memory=parameters["ray_memory"],
    )