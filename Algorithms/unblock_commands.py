

def main(gesture_seq, coords_seq, command_variables):
    if command_variables.get('is_blocked', True):
        command_variables['is_blocked'] = False
        print('commands available!')
    return command_variables