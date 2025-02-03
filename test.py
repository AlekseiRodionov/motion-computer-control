from GestureDetector import GestureDetector
from CommandExecutor import CommandExecutor
import json
import cv2

with open('config.json', 'r') as config_file:
    config_dict = json.load(config_file)

model1 = GestureDetector(model_type=config_dict['model_type'], path_to_file=config_dict['checkpoint_path'])
executor = CommandExecutor(config_dict['command_file_path'])
camera = cv2.VideoCapture(0)

gesture_seq = []
coords_seq = []

while True:
    _, image = camera.read()
    image = cv2.flip(image, 1)
    result = model1.predict(image)
    if result is not None:
        if len(gesture_seq) > 2:
            gesture_seq.pop(0)
            coords_seq.pop(0)
        gesture_seq.append(float(result['labels'][0]))
        coords_seq.append(result['boxes'][0])
        executor.execute_command(gesture_seq.copy(), coords_seq.copy())
    else:
        if len(gesture_seq) > 0:
            gesture_seq.pop(0)
            coords_seq.pop(0)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

del camera
