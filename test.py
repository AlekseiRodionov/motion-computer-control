from GestureDetector import GestureDetector
from CommandExecutor import CommandExecutor
import cv2

model1 = GestureDetector(model_type='YOLO', path_to_file='YOLOv10n_gestures.pt')
executor = CommandExecutor()
executor.create_command([17.0,], 'Algorithms/volume_mute.py')
executor.create_command([2.0,], 'Algorithms/block_commands.py')
executor.create_command([20.0,], 'Algorithms/mouse_move.py')
executor.create_command([14.0,], 'Algorithms/mouse_click.py')
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
        if len(gesture_seq) > 2:
            gesture_seq.pop(0)
            coords_seq.pop(0)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

del camera
