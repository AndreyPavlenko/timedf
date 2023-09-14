Adding new benchmark
====================

There is an easy way to add your own benchmark to our suit.

Each benchmark consists of following steps:

#. Receive execution parameters, such as path to a dataset, backend parameters, database parameters.
#. Run benchmark payload and measure timings in the process
#. Submit benchmark timings along with supportive information to a database

The library simplifies these steps for you, so you can focus on payload part.

Writing benchmark payload
-------------------------

``BenchmarkBase`` class is an interface for new benchmarks.
If you want to add new benchmark you need to:

1. Create directory inside of ``timedf_benchmarks`` folder and name it with your ``BENCHMARK_NAME``. You will store new benchmark's code in that directory and won't need to change anything else
2. Write your payload however you want inside of this directory, here is a minimal example with comments, this code is expected to be stored in ``timedf_benchmarks/example_benchmark/benchmark_content.py``

.. literalinclude:: ../../timedf_benchmarks/example_benchmark/benchmark_content.py
    :language: python
    :linenos:

3. Add ``__init__.py``  file along with benchmark_content  file:

.. literalinclude:: ../../timedf_benchmarks/example_benchmark/__init__.py
    :language: python
    :linenos:

Running new benchmark
---------------------

How to run
^^^^^^^^^^

You've modified library source code, so now you need to install it once again (or simply use ``pip install -e ".[all]"`` to use your folder version).

You can run your new benchmark with::

    benchmark-run -b example_benchmark -data_file "/MY_DATASET_PATH"


To see all available params run::

    benchmark-run -h

Benchmark run and iterations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each benchmark **run** consists of one or several **iterations** to
account for random effects.

You can set number of iterations with ``-iterations N`` parameter::

    benchmark-run ... -iterations N

Your payload will be run ``N`` times and results of each iteration will be recorded separately. So during analysis you will be
able to aggregate results however you want (min, max, median, mean).


Database connection
^^^^^^^^^^^^^^^^^^^

You need to have SQL database to store benchmark results.
If you don't have one, you can just use sqlite database that is essentialy just file on your filesystem.

Any database supported by ``sqlalchemy==1.4`` can be used,
so you can use most of databases, including MySQL, PostgreSQL, sqlite.

You set DB connection configuration with console arguments to ``benchmark-run``::

    benchmark-run example_benchmark \
        -data_file "/MY_DATASET_PATH" \
        -db_driver mysql+mysqlconnector \
        -db_server my_server.com \
        -db_port 3306 \
        -db_user my_username  \
        -db_pass my_pass  \
        -db_name my_database_name \

To save results to sqlite database (essentially just file ``db.sqlite``)::

    benchmark-run example_benchmark \
        -data_file "/MY_DATASET_PATH" \
        -db_driver sqlite+pysqlite \
        -db_name db.sqlite \

Benchmark result storage
------------------------

Each benchmark returns 2 results::

    # Measurements for benchmark steps
    measurement2time: Dict[str, float] = {'step1': 1.2, 'step2': 2.2}
    # Additional information that you want to submit along with benchmark results
    benchmark_specific_record: Union[Dict[str, str], None] = {'dataset_size': 1024}

These results are stored in SQL database in 2 tables:

iteration
    one row per benchmark iteration (single run of benchmark can contain several iterations

=================  ========   ===================  ==================================================================
Column             Type       Example              Description
=================  ========   ===================  ==================================================================
id                 int        122                  Primary key, unique for iteration
benchmark          varchar    plasticc             Benchmark name
run_id             int        1234123              Unique run id for this run
iteration_no       int        3                    Iteration number in it's run
date               datetime   2023-01-11 11:10:35  Date
server_name        varchar    my_laptop            Host name
...                ...        ...                  ...
*other host info*  ...        ...                  *Additional host info such as RAM, CPU, architecture. Generated automatically*
data_file          varchar    '/mnt/datasets`      Path to data file that was provided from console
...                varchar    ...                  *Other console arguments, saved automatically*
backend        varchar    Modin_on_hdk         Pandas mode, used for this run
...                varchar    ...                  *Other console arguments, saved automatically*
tag                varchar    new_patch            *Optional* tag for this run. Useful to identify unique runs, like experimental library version.
params             json       {'n_rows': 4000000}  Parameters passed by benchmark author using ``benchmark_specific_record``
=================  ========   ===================  ==================================================================

measurement
    one row per each measurement, so one iteration have many measurements

=================  ========   ===================  ==================================================================
Column             Type       Example              Description
=================  ========   ===================  ==================================================================
id                 int        1444                 Primary key for this table
name               varchar    load_data            Name of the measurment
duration_s         float      12.11                Duration of this measurement in seconds
iteration_id       int        122                  Foreign key, primary key for iteration
params             json       --                   *Additional fields, not used right now*
=================  ========   ===================  ==================================================================
