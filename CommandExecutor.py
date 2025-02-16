import json
from importlib.machinery import SourceFileLoader


class CommandExecutor:
    """
    This class allows you to establish a correspondence between gestures and algorithms (create commands),
    execute them, delete them, etc.

    Attributes:
        commands_dict (dict): A dictionary mapping gestures (keys) to python-script paths (values).
        commands_variables (dict): A dictionary (initially empty) intended for use by the user in Python scripts.
                                   It is passed to and returned from all called scripts. This allows the user to
                                   organize the exchange of information between different scripts.
    """

    def __init__(self):
        self.commands_dict = dict()
        self.commands_variables = dict()

    def load_commands_dict(self, path_to_commands_file: str):
        """
        The method loads a previously saved command dictionary from the specified file.

        Args:
            path_to_commands_file (str): Path to the json-file from which the command dictionary is loaded.

        Returns:
            None.
        """
        with open(path_to_commands_file, 'r', encoding='utf8') as commands_file:
            data = commands_file.read()
            if data:
                self.commands_dict = json.loads(data)
            else:
                self.commands_dict = dict()

    def create_command(self, path_to_commands_file: str, gesture: str, algorithm_path: str):
        """
        The method allows you to create a connection between a gesture and a python-script
        executed in response to this gesture, and also write this connection to the specified file.

        Args:
            path_to_commands_file (str): Path to the file where the created connection will be written.
            gesture (str): Gesture recognized in the image.
            algorithm_path (str): Path to the executable script.

        Returns:
            None.
        """
        self.commands_dict[gesture] = algorithm_path
        with open(path_to_commands_file, 'w', encoding='utf8') as commands_file:
            json.dump(self.commands_dict, commands_file)
    
    def delete_command(self, path_to_commands_file: str, gesture: str):
        """
        The method removes the specified gesture (and the corresponding path to the executable script)
        from the specified command file.

        Args:
            path_to_commands_file (str): Path to the file with commands.
            gesture (str): The gesture with which the command needs to be deleted.

        Returns:
            None.
        """
        with open(path_to_commands_file, 'r', encoding='utf8') as commands_file:
            data = commands_file.read()
            if data:
                self.commands_dict = json.loads(data)
                self.commands_dict.pop(gesture)
        with open(path_to_commands_file, 'w', encoding='utf8') as commands_file:
            json.dump(self.commands_dict, commands_file)

    def execute_command(self, gesture: str, coords: tuple[float, float, float, float]):
        """
        The method executes the algorithm corresponding to the received gesture.
        Note: in order for the algorithm to be executed, the Python script associated with the gesture
        must have a function named main (this is the one that is executed). This function must accept three
        arguments - the gesture itself, its coordinates, and the commands_variables dictionary.
        In addition, it must return the commands_variables dictionary. The user can change the
        commands_variables dictionary at his own discretion and thus pass information between different scripts.

        Args:
            gesture (str): A gesture for which a Python script was previously associated (in the create_command method).
            coords (tuple): Coordinates of the gesture recognized in the image (in "xywh" format by default).

        Returns:
            None.
        """
        algorithm_path = self.commands_dict.get(gesture)
        if algorithm_path is not None:
            algorithm = SourceFileLoader('algorithm', algorithm_path).load_module()
            self.commands_variables = algorithm.main(gesture, coords, self.commands_variables)
