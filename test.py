from GestureDetector import GestureDetector
import cv2

model = GestureDetector(model_type='YOLO', path_to_file='YOLOv10n_gestures.pt')
model2 = GestureDetector(model_type='SSDLite', path_to_file='SSDLiteMobileNetV3Large.pth')
camera = cv2.VideoCapture(0)

while True:
    _, image = camera.read()
    image = cv2.flip(image, 1)
    result = model.predict(image)
    result2 = model2.predict(image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

del camera
