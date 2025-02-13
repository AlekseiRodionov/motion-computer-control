from GestureDetector import GestureDetector
from CommandExecutor import CommandExecutor
import json
import cv2

with open('config.json', 'r') as config_file:
    config_dict = json.load(config_file)

model = GestureDetector(model_type=config_dict['model_type'], path_to_file=config_dict['checkpoint_path'])
executor = CommandExecutor()
executor.load_commands_dict(config_dict['command_file_path'])
camera = cv2.VideoCapture(config_dict['camera_index'])

while True:
    _, image = camera.read()
    image = cv2.flip(image, 1)
    result = model.predict(image, conf=float(config_dict['conf']), iou=float(config_dict['iou']))
    if result is not None:
        for box, label in zip(result['boxes'], result['labels']):
            gesture = int(label)
            coords = box
            executor.execute_command(gesture, coords)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

del camera
