from ultralytics import YOLO
import os

model = YOLO(os.path.join("checkpoints", "YOLOv10n_gestures.pt"))
model.export(format='onnx')