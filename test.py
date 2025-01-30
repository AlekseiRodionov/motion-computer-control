from GestureDetector import GestureDetector
from CommandExecutor import CommandExecutor
import cv2

model1 = GestureDetector(model_type='YOLO', path_to_file='YOLOv10n_gestures.pt')
executor = CommandExecutor()
executor.create_command([13.0,], 'Algorithms/block_commands.py') # 13.0 = "Dislike"
executor.create_command([16.0,], 'Algorithms/unblock_commands.py') # 16.0 = "Like"
executor.create_command([7.0,], 'Algorithms/close_program.py') # 7.0 = "Xsign"
executor.create_command([20.0,], 'Algorithms/hello_on_palm.py') # 20.0 = "Palm"
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
