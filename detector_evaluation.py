import json
import os

import cv2

from GestureDetector import GestureDetector


def detector_evaluation():
    """
    Model testing function. A separate video stream is opened, in which bounding boxes and
    class names are superimposed on frames. This function has no arguments - it reads all
    the necessary settings from the config-file, which is generated in MotionControlApp.py.
    """
    with open(os.path.join('configs', 'test_config.json'), 'r', encoding='utf8') as train_config_file:
        config = json.loads(train_config_file.read())
    model = GestureDetector(config['model_type'], config['path_to_checkpoint'])
    print('Включение камеры...')
    camera = cv2.VideoCapture(int(config['camera_index']))
    print('Камера включена. Чтобы закончить тестирование, закройте это окно.')
    while True:
        return_value, image = camera.read()
        image = cv2.flip(image, 1)
        result = model.predict(image, conf=float(config['conf']), iou=float(config['iou']), coords_format='x1y1x2y2')
        if result:
            for gesture, box in zip(result['labels'], result['boxes']):
                image = cv2.rectangle(image, (int(box[0]), int(box[1])),
                                      (int(box[2]), int(box[3])), (255, 0, 0), thickness=2)
                image = cv2.putText(image, gesture, (int(box[0]) + 4, int(box[1]) + 18), cv2.FONT_HERSHEY_SIMPLEX,
                                    fontScale=0.8, color=(255, 0, 0), thickness=2)
        cv2.imshow('test', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    del camera


detector_evaluation()
