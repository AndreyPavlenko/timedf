import os
import time

from timedf.timer import tm
from timedf.arg_parser import parse_args, prepare_general_parser
from timedf.benchmark import BaseBenchmark, create_benchmark
from timedf.backend import Backend


def make_benchmark() -> BaseBenchmark:
    """Function imports and creates benchmark object without using benchmark-specific arguments"""
    parser = prepare_general_parser()
    args = parser.parse_known_args()[0]
    bench_name = args.bench_name

    # Set current backend, !!!needs to be run before benchmark import!!!
    Backend.init(
        backend_name=args.backend,
        ray_tmpdir=args.ray_tmpdir,
        ray_memory=args.ray_memory,
        num_threads=args.num_threads,
    )

    return create_benchmark(bench_name)


def main():
    os.environ["PYTHONIOENCODING"] = "UTF-8"
    os.environ["PYTHONUNBUFFERED"] = "1"

    benchmark = make_benchmark()
    args, db_config = parse_args(benchmark.add_benchmark_args)

    data_file = args.data_file.replace("'", "")
    # Set verbosity level for global object, shared with benchmarks
    tm.verbosity = args.verbosity

    run_parameters = {
        "data_file": data_file,
        "backend": args.backend,
        "no_ml": args.no_ml,
        "num_threads": args.num_threads,
        # Used in ny_taxi_ml and plasticc
        "use_modin_xgb": args.use_modin_xgb,
        # Add benchmark-specific arguments
        **{k: getattr(args, k) for k in benchmark.__params__},
    }

    report_args = {
        "ray_tmpdir": args.ray_tmpdir,
        "ray_memory": args.ray_memory,
        "tag": args.tag,
    }

    benchmarkDb = db_config.maybeCreateBenchmarkDb()

    run_id = int(round(time.time()))
    print(run_parameters)

    for iter_num in range(1, args.iterations + 1):
        print(f"Iteration #{iter_num}")
        results = benchmark.run(run_parameters)
        tm.reset()

        if benchmarkDb is not None:
            benchmarkDb.report(
                iteration_no=iter_num,
                name2time=results.measurements,
                params=results.params,
                benchmark=args.save_benchmark_name or args.bench_name,
                backend=args.save_backend_name or args.backend,
                run_id=run_id,
                run_params={**run_parameters, **report_args},
            )


if __name__ == "__main__":
    main()
