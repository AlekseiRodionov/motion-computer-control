

def main(gesture_seq, coords_seq, command_executor):
    if command_executor.commands_variables.get('is_blocked', True):
        command_executor.commands_variables['is_blocked'] = False
        print('commands available!')