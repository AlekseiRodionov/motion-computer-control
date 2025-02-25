import os

def main(gesture_seq, coords_seq, command_executor):
    if not command_executor.commands_variables.get('is_blocked', False):
        print('hello!')
        try:
        	# A new command dictionary can be loaded in a similar manner.
        	command_executor.load_commands_dict(os.path.join('algorithms', 'examples', 'test_commands2.json'))
        except:
        	pass