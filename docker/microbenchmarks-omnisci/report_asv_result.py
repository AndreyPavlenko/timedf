import json
import sys
import os
import itertools as it
import math
import argparse
import mysql.connector


def get_db_parameters():
    return tuple()

# Reporting DB data
db_server, db_port, db_user, db_pass, db_name = get_db_parameters()

db_table = "25_modin_funcs"

results = []

class DbReport:
    """Initialize and submit reports to MySQL database"""

    def __init__(self, database, table_name, persistent_values=None):
        self.all_fields = {}
        self._table_name = table_name
        self._database = database

        if persistent_values:
            self.all_fields.update(persistent_values)

    def __quote_string(self, n):
        if type(n) is str:
            return "'" + n + "'"
        elif type(n) is float:
            if n == float("inf"):
                return "4294967295"
        return str(n)

    def submit(self, submit_values):
        def submit_row(row_values):
            self.sql_statement = f"INSERT INTO {self._table_name} ("
            self.all_fields.update(row_values)

            for n in list(self.all_fields.keys())[:-1]:
                self.sql_statement += n + ","
            self.sql_statement += list(self.all_fields)[-1] + ") VALUES("
            for n in list(self.all_fields.values())[:-1]:
                self.sql_statement += self.__quote_string(n) + ","
            n = list(self.all_fields.values())[-1]
            self.sql_statement += self.__quote_string(n) + ");"
            self.sql_statement = self.sql_statement.replace("'NULL'", "NULL")
            self._database.cursor().execute(self.sql_statement)

        if isinstance(submit_values, (list, set)):
            for row_values in submit_values:
                submit_row(row_values)
        elif isinstance(submit_values, dict):
            submit_row(submit_values)
        else:
            raise TypeError(f"Unsupported `submit_values` type: {type(submit_values)}")

        self._database.commit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-path",
        dest="result_path",
        type=str,
        required=True,
        default=None,
        help="File path of ASV result to report.",
    )
    result_path = parser.parse_args().result_path

    with open(result_path, "r") as f:
        res = json.load(f)

    reporting_init_fields = {
        "ServerName": res["params"]["machine"],
        "Machine": res["params"]["arch"],
        "CPUModel": res["params"]["cpu"],
        "CPUCount": res["params"]["num_cpu"],
        "OS": res["params"]["os"],
        "RAM": res["params"]["ram"],
        "ModinCommitHash": res["commit_hash"],
    }

    for benchmark in res["results"]:
        counter = 0
        bench_result = res["results"][benchmark]
        nested_lists = isinstance(bench_result, dict)
        params = (
            res["results"][benchmark]["params"]
            if nested_lists
            else res["results"][benchmark][1]
        )
        combinations = list(it.product(*params))
        test = ["_".join([str(y) for y in x]) for x in combinations]
        result_field = bench_result["result"] if nested_lists else bench_result[0]
        for comb in test:
            comb_result = (
                result_field[counter]
                if result_field is not None and result_field[counter] is not None
                else "NULL"
            )
            comb_result = (
                "NULL"
                if isinstance(comb_result, float) and math.isnan(comb_result)
                else comb_result
            )
            results.append(
                {
                    "Benchmark": benchmark,
                    "Parameters": comb.replace("'", ""),
                    "Result": comb_result,
                }
            )
            counter += 1

    db = mysql.connector.connect(
        host=db_server,
        port=db_port,
        user=db_user,
        passwd=db_pass,
        db=db_name,
    )

    db_reporter = DbReport(
        db,
        db_table,
        reporting_init_fields,
    )

    db_reporter.submit(results)

    print("Data was successfully reported!")
