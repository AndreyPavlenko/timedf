import os
import re
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils import execute_process


class CondaEnvironment:
    "Manage conda environments(create, remove, etc.)"

    def __init__(self, name):
        self.name = name

    def is_env_exist(self, name=None):
        env_name = name if name else self.name
        envs_list_cmdline = ['conda',
                             'env',
                             'list']
        _, output = execute_process(envs_list_cmdline)
        envs = re.findall(r'[^\s]+', output)
        if env_name in envs:
            return True
        return False

    def remove(self, name=None):
        env_name = name if name else self.name
        remove_env_cmdline = ['conda',
                              'env',
                              'remove',
                              '--name', env_name]
        execute_process(remove_env_cmdline)

    def create(self, existence_check=False, name=None, requirements_file=None):
        env_name = name if name else self.name
        if self.is_env_exist(env_name):
            if existence_check:
                return
            else:
                self.remove(env_name)
        create_env_cmdline = ['conda',
                              'env',
                              'create',
                              '--name', env_name,
                              f'--file={requirements_file}' if requirements_file else '']
        execute_process(create_env_cmdline, print_output=False)

    def add_conda_execution(self, cmdline, name=None):
        env_name = name if name else self.name
        cmd_res = ['conda', 'run', '-n', env_name]
        cmd_res.extend(cmdline)
        return cmd_res

