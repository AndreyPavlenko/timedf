import glob
import os
import pandas as pd
import pathlib
import signal
import sys
import subprocess
import threading
import time

path_to_ibis_dir = os.path.join(pathlib.Path(__file__).parent.parent, "..", "ibis/build/lib")
sys.path.insert(1, path_to_ibis_dir)
import ibis


class Omnisci_server:
    "Manage data directory, launch/termination and connection establishing with OmniSci server."

    _http_port = 62278
    _calcite_port = 62279
    _server_process = None
    _columns_names_santander_train = ["ID_code", "target"] + ["var_" + str(index) for index in range(200)]
    _header_santander_train = False

    def __init__(self, omnisci_executable, omnisci_port, database_name, table_name, omnisci_cwd=None):
        if omnisci_cwd is not None:
            self._server_cwd = omnisci_cwd
        else:
            self._server_cwd = pathlib.Path(omnisci_executable).parent.parent

        self._data_dir = os.path.join(self._server_cwd, "data")
        if not os.path.isdir(self._data_dir):
            print("CREATING DATA DIR", self._data_dir)
            os.makedirs(self._data_dir)
        if not os.path.isdir(os.path.join(self._data_dir, "mapd_data")):
            print("INITIALIZING DATA DIR", self._data_dir)
            self._initdb_executable = os.path.join(pathlib.Path(omnisci_executable).parent, "initdb")
            _ = self._execute_process([self._initdb_executable, '-f', '--data', self._data_dir])

        self._server_port = omnisci_port
        self._database_name = database_name
        self._table_name = table_name
        self._omnisci_server_executable = os.path.join(pathlib.Path(omnisci_executable).parent, "omnisci_server")
        self._server_start_cmdline = [self._omnisci_server_executable,
                                    "data",
                                    '--port', str(omnisci_port),
                                    '--http-port', str(self._http_port),
                                    '--calcite-port', str(self._calcite_port),
                                    '--config', "omnisci.conf"]
        
        self._omnisci_cmd_line = [omnisci_executable] + [str(self._database_name), "-u", "admin", "-p", "HyperInteractive"] + ["--port", str(self._server_port)]
        self._command_2_import_CSV = "COPY " + self._table_name + " FROM '%s' WITH (header='%s');"
        self._conn = None

    def _execute_process(self, cmdline, cwd=None):
        "Execute cmdline in user-defined directory by creating separated process"

        try:
            process = subprocess.Popen(cmdline, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            if process.returncode != 0 and process.returncode != None:
                raise Exception("Command returned {}".format(process.returncode))
        except OSError as err:
            print("Failed to start", cmdline, err)

        return process

    def _read_csv_datafile(self, file_name, columns_names, header=None, compression_type='gzip'):
        print("READING DATAFILE", file_name)
        return pd.read_csv(file_name, compression=compression_type, header=header, names=columns_names)
    
    def launch(self):
        "Launch OmniSci server"

        print("LAUNCHING SERVER ...")
        self._server_process = self._execute_process(self._server_start_cmdline, cwd=self._server_cwd)
        print("SERVER IS LAUNCHED")

    def terminate(self):
        "Terminate OmniSci server"

        print("TERMINATING SERVER ...")

        try:
            self._server_process.send_signal(signal.SIGINT)
            time.sleep(2)
            self._server_process.kill()
            time.sleep(1)
            self._server_process.terminate()
        except Exception as err:
            print("Failed to terminate server, error occured:", err)
            sys.exit(2)

        print("SERVER IS TERMINATED")

    def import_data(self, data_files_names, files_limit, header=False):
        "Import CSV files using COPY SQL statement"

        if header == True:
            header_value = 'true'
        elif header == False:
            header_value = 'false'
        else:
            print("Wrong value of header argument!")
            sys.exit(3)

        for f in data_files_names[:files_limit]:
            print("Importing datafile", f)
            copy_str = self._command_2_import_CSV % (f, header_value)

            try:
                import_process = subprocess.Popen(self._omnisci_cmd_line, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.PIPE)
                output = import_process.communicate(copy_str.encode())
            except OSError as err:
                print("Failed to start", self._omnisci_cmd_line, err)

            print(str(output[0].strip().decode()))
            print("Command returned", import_process.returncode)

    def import_data_by_ibis(self, data_files_names, files_limit, columns_names, header=None):
        "Import CSV files using Ibis load_data from the Pandas.DataFrame"

        t0 = time.time()
        pandas_df_from_each_file = (self._read_csv_datafile(file_name, columns_names, header) for file_name in data_files_names[:files_limit])
        pandas_concatenated_df = pd.concat(pandas_df_from_each_file, ignore_index=True)
        t_import_pandas = time.time() - t0

        t0 = time.time()
        self._conn.load_data(table_name=self._table_name, obj=pandas_concatenated_df, database=self._database_name)
        t_import_ibis = time.time() - t0

        return t_import_pandas, t_import_ibis

    def connect_to_server(self):
        "Connect to Omnisci server using Ibis framework"
        self._conn = ibis.omniscidb.connect(host="localhost", port=self._server_port, user="admin", password="HyperInteractive")
        return self._conn
