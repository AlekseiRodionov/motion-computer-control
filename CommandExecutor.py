import os
from importlib.machinery import SourceFileLoader
import json

class CommandExecutor():

    def __init__(self, path_to_commands_file='commands.json'):
        self.path_to_commands_file = path_to_commands_file
        self.commands_dict = self.__import_commands_dict()
        self.command_variables = dict()

    def __import_commands_dict(self):
        if not os.path.isfile(self.path_to_commands_file):
            return dict()
        with open(self.path_to_commands_file, 'r') as commands_file:
            commands_dict = json.load(commands_file)
        return commands_dict

    def create_command(self, gesture_seq=None, algorithm_filename=None):
        self.commands_dict[str(gesture_seq)] = algorithm_filename
        with open(self.path_to_commands_file, 'w') as commands_file:
            json.dump(self.commands_dict, commands_file)

    def __command_filtering(self, gesture_seq=None):
        while len(gesture_seq) > 0:
            str_gesture_seq = str(gesture_seq)
            if self.commands_dict.get(str_gesture_seq) is None:
                gesture_seq.pop(0)
            else:
                return self.commands_dict[str_gesture_seq]
        return None

    def execute_command(self, gesture_seq=None, coords_seq=None):
        # gesture_seq - это кортеж, внутри которого находятся классы жестов.
        # В будущем нужно будет придумать, как к этому добавить ещё координаты.
        algorithm_filename = self.__command_filtering(gesture_seq)
        if algorithm_filename is not None:
            algorithm = SourceFileLoader('algorithm', algorithm_filename).load_module()
            self.command_variables = algorithm.main(gesture_seq, coords_seq, self.command_variables)




