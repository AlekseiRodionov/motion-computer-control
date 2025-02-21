import os

def main(gesture_seq, coords_seq, command_executor):
    if not command_executor.commands_variables.get('is_blocked', False):
        print('hello!')
        command_executor.load_commands_dict(os.path.join('algorithms', 'examples', 'test_commands2.json'))