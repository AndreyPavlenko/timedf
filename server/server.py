import glob
import ibis
import os
import pathlib
import signal
import sys
import subprocess
import time


class OmnisciServer:
    "Manage data directory and launch/termination of the OmniSci server."

    def __init__(self, omnisci_executable, omnisci_port, omnisci_cwd=None):
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
            self._execute_process([self._initdb_executable, '-f', '--data', self._data_dir])

        self._server_port = omnisci_port
        self._server_process = None
        self._server_cmdline = [omnisci_executable,
                        'data',
                        '--port', str(omnisci_port),
                        '--http-port', "62278",
                        '--calcite-port', "62279",
                        '--config', 'omnisci.conf']
        
        self._command2ImportCSV = "COPY trips FROM '%s' WITH (header='false');"
        self._omnisciCmdLine = [omnisci_executable] + ["-q", "omnisci", "-u", "admin", "-p", "HyperInteractive"] + ["--port", str(omnisci_port)]

    def _execute_process(self, cmdline, cwd=None):
        try:
            process = subprocess.Popen(cmdline, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            out = process.communicate()[0].strip().decode()
            print(out)
        except OSError as err:
            print("Failed to start", cmdline, err)
        if process.returncode != 0:
            raise Exception("Command returned {}".format(process.returncode))
    
    def launch(self):
        print("LAUNCHING SERVER ...")
        try:
            nonlocal _server_process
            _server_process = subprocess.Popen(self._server_cmdline, cwd=self._server_cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        except Exception as err:
            print("Failed to launch server, error occured:", err)
            sys.exit(1)
        
        print("SERVER IS LAUNCHED")

    def terminate(self):
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

    def import_data(self, dataFileNames, files_limit):
        "Import CSV files using COPY"

        for f in dataFileNames[:files_limit]:
            print("Importing datafile", f)
            copyStr = self._command2ImportCSV % f
            try:
                process = subprocess.Popen(self._omnisciCmdLine, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.PIPE)
                output = process.communicate(copyStr.encode())
            except OSError as err:
                print("Failed to start", self._omnisciCmdLine, err)

            print(str(output[0].strip().decode()))
            print("Command returned", process.returncode)

    def connect_to_server(self):
        return ibis.omniscidb.connect(host="localhost", port=self._server_port, user="admin", password="HyperInteractive")
