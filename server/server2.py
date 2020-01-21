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
    "Manage interactions with OmniSciDB server (launch/termination, connection establishing, etc.)"

    _http_port = 62278
    _calcite_port = 62279
    server_process = None
    _header_santander_train = False
    _imported_pd_df = {}

    def __init__(self, omnisci_executable, omnisci_port, database_name, omnisci_cwd=None):
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
        self._omnisci_server_executable = os.path.join(pathlib.Path(omnisci_executable).parent, "omnisci_server")
        self._server_start_cmdline = [self._omnisci_server_executable,
                                    "data",
                                    '--port', str(omnisci_port),
                                    '--http-port', str(self._http_port),
                                    '--calcite-port', str(self._calcite_port),
                                    '--config', "omnisci.conf"]
        
        self._omnisci_cmd_line = [omnisci_executable] + [str(self._database_name), "-u", "admin", "-p", "HyperInteractive"] + ["--port", str(self._server_port)]
        self._command_2_import_CSV = "COPY %s FROM '%s' WITH (header='%s');"
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

    def _read_csv_datafile(self, file_name, columns_names, header=None, compression_type='gzip', nrows=200000):
        "Read csv by Pandas. Function returns Pandas DataFrame, which can be used by ibis load_data function"
        
        print("Reading datafile", file_name)
        return pd.read_csv(file_name, compression=compression_type, header=header, names=columns_names, nrows=nrows)
    
    def connect_to_server(self):
        "Connect to Omnisci server using Ibis framework"
        
        self._conn = ibis.omniscidb.connect(host="localhost", port=self._server_port, user="admin", password="HyperInteractive")
        return self._conn

    def launch(self):
        "Launch OmniSciDB server"

        print("Launching server ...")
        self.server_process = self._execute_process(self._server_start_cmdline, cwd=self._server_cwd)
        print("Server is launched")

    def terminate(self):
        "Terminate OmniSci server"

        print("Terminating server ...")

        try:
            #self._conn.close()
            self.server_process.send_signal(signal.SIGINT)
            time.sleep(2)
            self.server_process.kill()
            time.sleep(1)
            self.server_process.terminate()
        except Exception as err:
            print("Failed to terminate server, error occured:", err)
            sys.exit(1)

        print("Server is terminated")

    def import_data(self, table_name, data_files_names, files_limit, header=False):
        "Import CSV files using COPY SQL statement"

        if header == True:
            header_value = 'true'
        elif header == False:
            header_value = 'false'
        else:
            print("Wrong value of header argument!")
            sys.exit(2)

        for f in data_files_names[:files_limit]:
            print("Importing datafile", f)
            copy_str = self._command_2_import_CSV % (table_name, f, header_value)

            try:
                import_process = subprocess.Popen(self._omnisci_cmd_line, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.PIPE)
                output = import_process.communicate(copy_str.encode())
            except OSError as err:
                print("Failed to start", self._omnisci_cmd_line, err)

            print(str(output[0].strip().decode()))
            print("Command returned", import_process.returncode)
    
    def import_data_by_ibis(self, table_name, data_files_names, files_limit, columns_names, columns_types, cast_dict, header=None):
        "Import CSV files using Ibis load_data from the Pandas.DataFrame"
        
        schema_table = ibis.Schema(
            names = columns_names,
            types = columns_types
        )
        
        if not self._conn.exists_table(name=table_name, database=self._database_name):
            try:
                self._conn.create_table(table_name = table_name, schema=schema_table, database=self._database_name)
            except Exception as err:
                print("Failed to create table:", err)

        t0 = time.time()
        if files_limit > 1:
            pandas_df_from_each_file = (self._read_csv_datafile(file_name, columns_names, header) for file_name in data_files_names[:files_limit])
            pandas_concatenated_df = pd.concat(pandas_df_from_each_file, ignore_index=True)
        else:
            pandas_concatenated_df = self._read_csv_datafile(data_files_names, columns_names, header)
        
        t_import_pandas = time.time() - t0

        if table_name not in self._imported_pd_df:
            self._imported_pd_df[table_name] = pandas_concatenated_df.astype(dtype=cast_dict, copy=False)

        t0 = time.time()
        self._conn.load_data(table_name=table_name, obj=self._imported_pd_df[table_name], database=self._database_name)
        t_import_ibis = time.time() - t0

        return t_import_pandas, t_import_ibis
    
    def drop_table(self, table_name):
        "Drop table by table_name using Ibis framework"
        
        if self._conn.exists_table(name=table_name, database=self._database_name):
            db = self._conn.database(self._database_name)
            df = db.table(table_name)
            df.drop()
            if table_name in self._imported_pd_df:
                del self._imported_pd_df[table_name]
        else:
            print("Table", table_name, "doesn't exist!")
            sys.exit(3)
    
    def get_pd_df(self, table_name):
        "Get already imported Pandas DataFrame"
        
        if self._conn.exists_table(name=table_name, database=self._database_name):
            return self._imported_pd_df[table_name]
        else:
            print("Table", table_name, "doesn't exist!")
            sys.exit(4)
