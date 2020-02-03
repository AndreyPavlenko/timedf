import os
import pathlib
import signal
import sys
import subprocess
import threading
import time


def execute_process(cmdline, cwd=None, shell=False, daemon=False, print_output=True):
    "Execute cmdline in user-defined directory by creating separated process"
    try:
        print("CMD: ", " ".join(cmdline))
        output = ""
        process = subprocess.Popen(cmdline, cwd=cwd, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, shell=shell)
        if not daemon:
            output = process.communicate()[0].strip().decode()
            if print_output:
                print(output)
        if process.returncode != 0 and process.returncode is not None:
            raise Exception(f"Command returned {process.returncode}. \n{output}")
    except OSError as err:
        print("Failed to start", cmdline, err)

    return process, output


class OmnisciServer:
    "Manage interactions with OmniSciDB server (launch/termination, connection establishing, etc.)"

    server_process = None
    _header_santander_train = False

    def __init__(self, omnisci_executable, omnisci_port, database_name,
                 omnisci_cwd=None, user="admin", password="HyperInteractive", http_port=62278,
                 calcite_port=62279):
        self.omnisci_executable = omnisci_executable
        self.server_port = omnisci_port
        self.user = user
        self.password = password
        self.database_name = database_name
        self._http_port = http_port
        self._calcite_port = calcite_port

        if omnisci_cwd is not None:
            self._server_cwd = omnisci_cwd
        else:
            self._server_cwd = pathlib.Path(self.omnisci_executable).parent.parent

        self._data_dir = os.path.join(self._server_cwd, "data")
        if not os.path.isdir(self._data_dir):
            print("CREATING DATA DIR", self._data_dir)
            os.makedirs(self._data_dir)
        if not os.path.isdir(os.path.join(self._data_dir, "mapd_data")):
            print("INITIALIZING DATA DIR", self._data_dir)
            self._initdb_executable = os.path.join(pathlib.Path(self.omnisci_executable).parent,
                                                   "initdb")
            execute_process([self._initdb_executable, '-f', '--data', self._data_dir])

        self.omnisci_sql_executable = os.path.join(pathlib.Path(self.omnisci_executable).parent,
                                                   "omnisql")
        self._server_start_cmdline = [self.omnisci_executable,
                                      "data",
                                      '--port', str(omnisci_port),
                                      '--http-port', str(self._http_port),
                                      '--calcite-port', str(self._calcite_port),
                                      '--config', "omnisci.conf",
                                      '--enable-watchdog=false',
                                      '--allow-cpu-retry']

    def launch(self):
        "Launch OmniSciDB server"

        print("Launching server ...")
        self.server_process, _ = execute_process(self._server_start_cmdline, cwd=self._server_cwd,
                                                 daemon=True)
        print("Server is launched")
        try:
            pt = threading.Thread(target=self._print_omnisci_output,
                                  args=(self.server_process.stdout,), daemon=True)
            pt.start()

            # Allow server to start up. It has to open TCP port and start
            # listening, otherwise the following benchmarks fail.
            time.sleep(5)
        except Exception as err:
            print("Failed", err)
            sys.exit(1)

    def _print_omnisci_output(self, stdout):
        for line in iter(stdout.readline, b''):
            print("OMNISCI>>", line.decode().strip())

    def terminate(self):
        "Terminate OmniSci server"

        print("Terminating server ...")

        try:
            if self.server_process:
                self.server_process.send_signal(signal.SIGINT)
                time.sleep(2)
                self.server_process.kill()
                time.sleep(1)
        except Exception as err:
            print("Failed to terminate server, error occured:", err)
            sys.exit(1)

        print("Server is terminated")
