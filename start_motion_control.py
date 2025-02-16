import json
import os

import cv2

from GestureDetector import GestureDetector
from CommandExecutor import CommandExecutor


def start_motion_control():
    """
    The function receives a video stream from the camera, detects gestures in the frame in real time
    and executes the commands associated with them. This function has no arguments - it reads all
    the necessary settings from the config-file, which is generated in MotionControlApp.py.
    """
    with open(os.path.join('configs', 'start_app_config.json'), 'r') as config_file:
        config_dict = json.load(config_file)
    print('Запуск программы...')
    model = GestureDetector(model_type=config_dict['model_type'], path_to_checkpoint=config_dict['path_to_checkpoint'])
    executor = CommandExecutor()
    executor.load_commands_dict(config_dict['command_file_path'])
    print('Включение камеры...')
    camera = cv2.VideoCapture(config_dict['camera_index'])
    print('Камера включена. Программа работает.')
    while True:
        _, image = camera.read()
        image = cv2.flip(image, 1)
        result = model.predict(image, conf=float(config_dict['conf']), iou=float(config_dict['iou']))
        if result is not None:
            for gesture, box in zip(result['labels'], result['boxes']):
                executor.execute_command(gesture, tuple(box))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    del camera


start_motion_control()
