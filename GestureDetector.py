from ultralytics import YOLO
import torch
from torchvision.transforms.functional import to_tensor
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from ssdlite_train.engine import train_one_epoch, evaluate
from ssdlite_train import utils
import os
import cv2

ALL_GESTURES = (
0, #'grabbing',
1, #'grip',
2, #'holy',
3, #'point',
4, #'call',
5, #'three3',
6, #'timeout',
7, #'xsign',
8, #'hand_heart',
9, #'hand_heart2',
10, #'little_finger',
11, #'middle_finger',
12, #'take_picture',
13, #'dislike',
14, #'fist',
15, #'four',
16, #'like',
17, #'mute',
18, #'ok',
19, #'one',
20, #'palm',
21, #'peace',
22, #'peace_inverted',
23, #'rock',
24, #'stop',
25, #'stop_inverted',
26, #'three',
27, #'three2',
28, #'two_up',
29, #'two_up_inverted',
30, #'three_gun',
31, #'thumb_index',
32, #'thumb_index2',
33, #'no_gesture'
)

class SSDLiteDataset(torch.utils.data.Dataset):
    def __init__(self, path_to_data, transforms):
        self.path_to_data = path_to_data
        self.transforms = transforms
        self.images = list(sorted(os.listdir(os.path.join(path_to_data, 'images'))))
        self.labels = list(sorted(os.listdir(os.path.join(path_to_data, 'labels'))))
        if len(self.images) % 2 != 0:
            self.images.append(self.images[0])
            self.labels.append(self.labels[0])
        self.image_id = 0

    def __getitem__(self, item):
        image_path = os.path.join(self.path_to_data, 'images', self.images[item])
        label_path = os.path.join(self.path_to_data, 'labels', self.labels[item])
        image = to_tensor(cv2.imread(image_path))
        image_size = image.size()[1:]
        with open(label_path, 'r', encoding='utf8') as label_file:
            labels_boxes_list = label_file.read().split('\n')[:-1]
            target = {
                'boxes': [],
                'labels': []
            }
            for label_box in labels_boxes_list:
                label = label_box.split(' ')[0]
                box = label_box.split(' ')[1:]
                box = list(map(lambda x: float(x), box))
                box[0] = int(box[0] * image_size[0])
                box[1] = int(box[1] * image_size[1])
                box[2] = int(box[2] * image_size[0])
                box[3] = int(box[3] * image_size[1])
                target['labels'].append(torch.tensor(float(label)))
                if self.transforms is not None:
                    box = self.transforms([box])[0]
                    new_box = torch.tensor([float(box[0]) if box[0] < box[2] else float(box[2]),
                                            float(box[1]) if box[1] < box[3] else float(box[3]),
                                            float(box[2]) if box[2] >= box[0] else float(box[0]),
                                            float(box[3]) if box[3] >= box[1] else float(box[1]),
                                            ])
                    target['boxes'].append(new_box)
                else:
                    target['boxes'].append(torch.tensor(box))
        target['boxes'] = torch.stack(target['boxes'], dim=0)
        target['labels'] = torch.stack(target['labels'], dim=0).long()
        return image, target

    def __len__(self):
        return len(self.images)


class GestureDetector():
    """
    This class is a universal interface for fundamentally different models.
    It is a wrapper for YOLO, SSDLite, and also the onnx-format of models.
    """

    def __init__(self, model_type='YOLO', path_to_file='YOLOv10n_gestures.pt'):
        """"""
        self.model_type = model_type
        if self.model_type == 'YOLO':
            self.model = YOLO(path_to_file)
        if self.model_type == 'SSDLite':
            self.model = ssdlite320_mobilenet_v3_large(pretrained=False, pretrained_backbone=False, num_classes=35)
            gesture_checkpoint = torch.load(path_to_file, map_location=torch.device('cpu'))
            self.model.load_state_dict(gesture_checkpoint['MODEL_STATE'])
            self.model.eval()
        if self.model_type == 'onnx':
            pass

    def __is_used_gesture(self, boxes_labels_dict, used_gestures):
        boxes = []
        labels = []
        for box, label in zip(boxes_labels_dict['boxes'], boxes_labels_dict['labels']):
            if label in used_gestures:
                boxes.append(box)
                labels.append(label)
        boxes_labels_dict = {'boxes': boxes, 'labels': labels}
        return boxes_labels_dict

    def __intersection(self, box1, box2):
        box1_x1, box1_y1, box1_x2, box1_y2 = box1[:4]
        box2_x1, box2_y1, box2_x2, box2_y2 = box2[:4]
        x1 = max(box1_x1, box2_x1)
        y1 = max(box1_y1, box2_y1)
        x2 = min(box1_x2, box2_x2)
        y2 = min(box1_y2, box2_y2)
        return (x2 - x1) * (y2 - y1)

    def __union(self, box1, box2):
        box1_x1, box1_y1, box1_x2, box1_y2 = box1[:4]
        box2_x1, box2_y1, box2_x2, box2_y2 = box2[:4]
        box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
        box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
        return box1_area + box2_area - self.__intersection(box1, box2)

    def __iou(self, box1, box2):
        return self.__intersection(box1, box2) / self.__union(box1, box2)

    def __non_maximum_supression(self, boxes, labels, iou):
        nms_boxes = []
        nms_labels = []
        while len(boxes) > 0:
            nms_boxes.append(boxes[0])
            nms_labels.append(labels[0])
            new_boxes = []
            new_labels = []
            for box, label in zip(boxes, labels):
                if self.__iou(box, boxes[0]) < iou:
                    new_boxes.append(box)
                    new_labels.append(label)
            boxes, labels = new_boxes, new_labels
        return nms_boxes, nms_labels

    def xywh_to_x1y1x2y2(self, boxes):
        x1y1x2y2_boxes = []
        for box in boxes:
            x1 = float(box[0] - box[2] / 2)
            y1 = float(box[1] + box[3] / 2)
            x2 = float(box[0] + box[2] / 2)
            y2 = float(box[1] - box[3] / 2)
            x1y1x2y2_boxes.append(torch.tensor([x1, y1, x2, y2]))
        return x1y1x2y2_boxes

    def x1y1x2y2_to_xywh(self, boxes):
        xywh_boxes = []
        for box in boxes:
            x_center = float(box[0] + box[2]) / 2
            y_center = float(box[1] + box[3]) / 2
            w = float(box[2] - box[0])
            h = float(box[3] - box[1])
            xywh_boxes.append(torch.tensor([x_center, y_center, w, h]))
        return xywh_boxes

    def predict(self, image, conf=0.5, iou=0.5, used_gestures=ALL_GESTURES, coords_format='xywh'):
        if self.model_type == 'YOLO':
            y_pred = self.model.predict(image, verbose=False, show=False, conf=conf, iou=iou)[0]
            if not y_pred.boxes:
                return None
            if coords_format == 'xywh':
                result = {'boxes': y_pred.boxes.xywh, 'labels': y_pred.boxes.cls}
            else:
                result = {'boxes': y_pred.boxes.xyxy, 'labels': y_pred.boxes.cls}
            result = self.__is_used_gesture(result, used_gestures)
        if self.model_type == 'SSDLite':
            image_tensor = to_tensor(image).unsqueeze(dim=0)
            y_pred = self.model(image_tensor)[0]
            y_pred['boxes'] = y_pred['boxes'][y_pred['scores'] > conf]
            y_pred['labels'] = y_pred['labels'][y_pred['scores'] > conf]
            if not y_pred['boxes'].size()[0]:
                return None
            if coords_format == 'xywh':
                y_pred['boxes'] = self.x1y1x2y2_to_xywh(y_pred['boxes'])
            y_pred = self.__is_used_gesture(y_pred, used_gestures)
            nms_boxes, nms_labels = self.__non_maximum_supression(y_pred['boxes'], y_pred['labels'], iou)
            result = {'boxes': nms_boxes, 'labels': nms_labels}
        if self.model_type == 'onnx':
            pass
        return result

    def fit(self, path_to_data, epochs, final_checkpoint_name, final_model_type):
        if self.model_type == 'YOLO':
            self.model.train(data=os.path.join(path_to_data, 'data.yaml'),
                             epochs=epochs,
                             imgsz=640,
                             freeze=20,
                             project=os.getcwd()
                             )
            self.model.save(final_checkpoint_name)

        if self.model_type == 'SSDLite':
            device = torch.device('cpu')
            dataset_generator = SSDLiteDataset(path_to_data, self.xywh_to_x1y1x2y2)
            dataloader = torch.utils.data.DataLoader(
                dataset_generator,
                batch_size=2,
                shuffle=False,
                collate_fn=utils.collate_fn
            )
            self.model.to(device)
            params = [p for p in self.model.parameters() if p.requires_grad]
            for i, p in enumerate(params):
                if i < len(params) - 36:
                    p.requires_grad = False
            optimizer = torch.optim.SGD(
                params,
                lr=0.005,
                momentum=0.9,
                weight_decay=0.0005
            )
            lr_sheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=3,
                gamma=0.1
            )

            for epoch in range(epochs):
                train_one_epoch(self.model, optimizer, dataloader, device, epoch, print_freq=10)
                lr_sheduler.step()
                checkpoint = {
                    'epoch': epoch,
                    'MODEL_STATE': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
            torch.save(checkpoint, final_checkpoint_name)

        print(f'Обучение завершено. Новый чекпойнт сохранён по указанному пути.\n')
        input('Нажмите Enter для завершения обучения.')