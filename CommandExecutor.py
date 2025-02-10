import os
from importlib.machinery import SourceFileLoader
import json

class CommandExecutor():

    def __init__(self):
        self.commands_dict = dict()
        self.command_variables = dict()

    def load_commands_dict(self, path_to_commands_file):
        with open(path_to_commands_file, 'r', encoding='utf8') as commands_file:
            data = commands_file.read()
            if data:
                self.commands_dict = json.loads(data)
            else:
                self.commands_dict = dict()
        return self.commands_dict

    def create_command(self, path_to_commands_file, gesture=None, algorithm_filename=None):
        self.commands_dict[str(gesture)] = algorithm_filename
        with open(path_to_commands_file, 'w', encoding='utf8') as commands_file:
            json.dump(self.commands_dict, commands_file)
    
    def delete_command(self, path_to_commands_file, gesture):
        with open(path_to_commands_file, 'r', encoding='utf8') as commands_file:
            data = commands_file.read()
            if data:
                self.commands_dict = json.loads(data)
                self.commands_dict.pop(str(gesture))
        with open(path_to_commands_file, 'w', encoding='utf8') as commands_file:
            json.dump(self.commands_dict, commands_file)

    def execute_command(self, gesture=None, coords=None):
        algorithm_filename = self.commands_dict.get(str(gesture))
        if algorithm_filename is not None:
            algorithm = SourceFileLoader('algorithm', algorithm_filename).load_module()
            self.command_variables = algorithm.main(gesture, coords, self.command_variables)




