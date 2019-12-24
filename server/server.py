import glob
import os
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
    _command_2_import_CSV = "COPY trips FROM '%s' WITH (header='false');"

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
        self._omnisci_server_executable = os.path.join(pathlib.Path(omnisci_executable).parent, "omnisci_server")
        self._server_start_cmdline = ["sudo", self._omnisci_server_executable,
                                    "data",
                                    '--port', str(omnisci_port),
                                    '--http-port', str(self._http_port),
                                    '--calcite-port', str(self._calcite_port),
                                    '--config', "omnisci.conf"]
        
        self._omnisci_cmd_line = [omnisci_executable] + [str(database_name), "-u", "admin", "-p", "HyperInteractive"] + ["--port", str(self._server_port)]

    def _execute_process(self, cmdline, cwd=None):
        "Execute cmdline in user-defined directory by creating separated process"

        try:
            process = subprocess.Popen(cmdline, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            if process.returncode != 0 and process.returncode != None:
                raise Exception("Command returned {}".format(process.returncode))
        except OSError as err:
            print("Failed to start", cmdline, err)

        return process
    
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

    def import_data(self, data_files_names, files_limit):
        "Import CSV files using COPY SQL statement"

        for f in data_files_names[:files_limit]:
            print("Importing datafile", f)
            copy_str = self._command_2_import_CSV % f

            try:
                import_process = subprocess.Popen(self._omnisci_cmd_line, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.PIPE)
                output = import_process.communicate(copy_str.encode())
            except OSError as err:
                print("Failed to start", self._omnisci_cmd_line, err)

            print(str(output[0].strip().decode()))
            print("Command returned", import_process.returncode)

    def connect_to_server(self):
        "Connect to Omnisci server using Ibis framework"

        return ibis.omniscidb.connect(host="localhost", port=self._server_port, user="admin", password="HyperInteractive")
