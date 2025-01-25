from GestureDetector import GestureDetector
import cv2

model1 = GestureDetector(model_type='YOLO', path_to_file='YOLOv10n_gestures.pt')
model2 = GestureDetector(model_type='SSDLite', path_to_file='SSDLiteMobileNetV3Large.pth')
camera = cv2.VideoCapture(0)

_, image = camera.read()
image = cv2.flip(image, 1)
print('model 1 result:\n ', model.predict(image))
print('model 2 result:\n ', model2.predict(image))

del camera
