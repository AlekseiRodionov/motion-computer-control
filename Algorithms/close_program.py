import sys

def main(gesture_seq, coords_seq, command_variables):
    if not command_variables.get('is_blocked', False):
        print('program is closed!')
        sys.exit()
    return command_variables