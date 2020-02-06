import argparse
import subprocess
import re


def str_arg_to_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Cannot recognize boolean value.')


def combinate_requirements(ibis, ci, res):
    with open(ibis) as f_ibis:
        ibis_data = f_ibis.read()
    with open(ci) as f_ci:
        ci_data = f_ci.read()
    with open(res, "w") as f_res:
        for line in ci_data.split('\n'):
            if line not in ibis_data:
                ibis_data += line + '\n'
        f_res.write(ibis_data)


def execute_process(cmdline, cwd=None, shell=False, daemon=False, print_output=True):
    "Execute cmdline in user-defined directory by creating separated process"
    try:
        print("CMD: ", " ".join(cmdline))
        output = ""
        process = subprocess.Popen(cmdline, cwd=cwd, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, shell=shell)
        if not daemon:
            output = process.communicate()[0].strip().decode()
            if re.findall(r"\d fail", output) or re.findall(r"[e,E]rror", output):
                process.returncode = 1
            elif print_output:
                print(output)
        if process.returncode != 0 and process.returncode is not None:
            raise Exception(f"Command returned {process.returncode}. \n{output}")
        return process, output
    except OSError as err:
        print("Failed to start", cmdline, err)
