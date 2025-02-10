import cv2
from GestureDetector import GestureDetector
import json

ALL_GESTURES = {
'0': 'grabbing',
'1': 'grip',
'2': 'holy',
'3': 'point',
'4': 'call',
'5': 'three3',
'6': 'timeout',
'7': 'xsign',
'8': 'hand_heart',
'9': 'hand_heart2',
'10': 'little_finger',
'11': 'middle_finger',
'12': 'take_picture',
'13': 'dislike',
'14': 'fist',
'15': 'four',
'16': 'like',
'17': 'mute',
'18': 'ok',
'19': 'one',
'20': 'palm',
'21': 'peace',
'22': 'peace_inverted',
'23': 'rock',
'24': 'stop',
'25': 'stop_inverted',
'26': 'three',
'27': 'three2',
'28': 'two_up',
'29': 'two_up_inverted',
'30': 'three_gun',
'31': 'thumb_index',
'32': 'thumb_index2',
'33': 'no_gesture'
}


with open('test_config.json', 'r', encoding='utf8') as train_config_file:
    config = json.loads(train_config_file.read())

model = GestureDetector(config['model_type'], config['path_to_init_checkpoint'])
camera = cv2.VideoCapture(int(config['camera_index']))

while True:
    return_value, image = camera.read()
    image = cv2.flip(image, 1)
    result = model.predict(image, conf=float(config['conf']), iou=float(config['iou']), coords_format='x1y1x2y2')
    if result:
        for box, label in zip(result['boxes'], result['labels']):
            image = cv2.rectangle(image, (int(box[0]), int(box[1])),
                                         (int(box[2]), int(box[3])), (255, 0, 0), thickness=2)
            image = cv2.putText(image, ALL_GESTURES[str(int(label))], (int(box[0]) + 4, int(box[1]) + 18), cv2.FONT_HERSHEY_SIMPLEX,
                                        fontScale=0.8, color=(255, 0, 0), thickness=2)
    cv2.imshow('video', image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

del camera