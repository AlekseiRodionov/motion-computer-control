import sys

def main(gesture_seq, coords_seq, command_executor):
    if not command_executor.commands_variables.get('is_blocked', False):
        print('program is closed!')
        sys.exit()