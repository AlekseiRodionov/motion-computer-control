

def main(gesture_seq, coords_seq, command_variables):
    if not command_variables.get('is_blocked', False):
        command_variables['is_blocked'] = True
        print('commands blocked!')
    return command_variables