import os

import cv2
import torch
from ultralytics import YOLO
from torchvision.transforms.functional import to_tensor
from torchvision.models.detection import ssdlite320_mobilenet_v3_large

from ssdlite_train_scripts.engine import train_one_epoch
from ssdlite_train_scripts import utils

NUM_TO_GESTURE = {
    0: 'grabbing',
    1: 'grip',
    2: 'holy',
    3: 'point',
    4: 'call',
    5: 'three3',
    6: 'timeout',
    7: 'xsign',
    8: 'hand_heart',
    9: 'hand_heart2',
    10: 'little_finger',
    11: 'middle_finger',
    12: 'take_picture',
    13: 'dislike',
    14: 'fist',
    15: 'four',
    16: 'like',
    17: 'mute',
    18: 'ok',
    19: 'one',
    20: 'palm',
    21: 'peace',
    22: 'peace_inverted',
    23: 'rock',
    24: 'stop',
    25: 'stop_inverted',
    26: 'three',
    27: 'three2',
    28: 'two_up',
    29: 'two_up_inverted',
    30: 'three_gun',
    31: 'thumb_index',
    32: 'thumb_index2',
    33: 'no_gesture'
}


class SSDLiteDataset(torch.utils.data.Dataset):
    """
    A dataset class designed to load data during training of the SDLite model.

    Attributes:
        path_to_data (str): Path to the folder from which data will be read.
        transforms (obj): A function designed to preprocess the coordinates of bounding boxes.
        images (list[str]): List of paths to images that will be loaded during training.
        labels (list[str]): List of paths to labels (object classes and their coordinates).

    Args:
        path_to_data (str): Path to the folder from which data will be read.
        transforms (obj): A function designed to preprocess the coordinates of bounding boxes.
    """

    def __init__(self, path_to_data, transforms):
        self.path_to_data = path_to_data
        self.transforms = transforms
        self.images = list(sorted(os.listdir(os.path.join(path_to_data, 'images'))))
        self.labels = list(sorted(os.listdir(os.path.join(path_to_data, 'labels'))))
        if len(self.images) % 2 != 0:
            self.images.append(self.images[0])
            self.labels.append(self.labels[0])

    def __getitem__(self, index):
        """
        The method returns the image by index, as well as the corresponding object labels
        (classes and coordinates of the bounding boxes).

        Args:
            index (int): training sample index.

        Returns:
            image (np.array[float]): An image represented as a numpy-array.
            targets_dict (dict[list]): Dictionary with object labels.
        """
        image_path = os.path.join(self.path_to_data, 'images', self.images[index])
        label_path = os.path.join(self.path_to_data, 'labels', self.labels[index])
        image = to_tensor(cv2.imread(image_path))
        image_size = image.size()[1:]
        with open(label_path, 'r', encoding='utf8') as label_file:
            targets_list = label_file.read().split('\n')
            targets_dict = {
                'boxes': [],
                'labels': []
            }
            for target in targets_list:
                if target:
                    label = target.split(' ')[0]
                    box = list(map(lambda x: float(x), target.split(' ')[1:]))
                    box[0] = int(box[0] * image_size[0])
                    box[1] = int(box[1] * image_size[1])
                    box[2] = int(box[2] * image_size[0])
                    box[3] = int(box[3] * image_size[1])
                    targets_dict['labels'].append(torch.tensor(float(label)))
                    if self.transforms is not None:
                        transformed_box = self.transforms([box])[0]
                        targets_dict['boxes'].append(transformed_box)
                    else:
                        targets_dict['boxes'].append(torch.tensor(box))
        targets_dict['boxes'] = torch.stack(targets_dict['boxes'], dim=0)
        targets_dict['labels'] = torch.stack(targets_dict['labels'], dim=0).long()
        return image, targets_dict

    def __len__(self):
        """
        The method returns the number of samples in the training set.
        """
        return len(self.images)


class GestureDetector:
    """
    This class is a universal interface for fundamentally different models.
    It is a wrapper for YOLO and SSDLite formats of models.

    Attributes:
        model_type (str): Model type: YOLO and SSDLite types are available.
        model (obj): Model. The neural network used.
        path_to_checkpoint (str): Path to the checkpoint from which the model is loaded.

    Args:
        model_type (str): Model type: YOLO and SSDLite types are available.
        path_to_checkpoint (str): Path to the checkpoint from which the model is loaded.
    """

    def __init__(self, model_type: str = 'YOLO', path_to_checkpoint: str = 'YOLOv10n_gestures.pt'):
        self.model_type = model_type
        if self.model_type == 'YOLO':
            self.model = YOLO(path_to_checkpoint)
        if self.model_type == 'SSDLite':
            self.model = ssdlite320_mobilenet_v3_large(pretrained=False, pretrained_backbone=False, num_classes=35)
            gesture_checkpoint = torch.load(path_to_checkpoint, map_location=torch.device('cpu'))
            self.model.load_state_dict(gesture_checkpoint['MODEL_STATE'])
            self.model.eval()

    def __intersection(self, box1: torch.tensor, box2: torch.tensor):
        """
        The method calculates the intersection between two bounding boxes.

        Args:
            box1 (torch.tensor[float]): First bounding box.
            box2 (torch.tensor[float]): Second bounding box.

        Returns:
            intersection (float): The amount of intersection between two bounding boxes.
        """
        box1_x1, box1_y1, box1_x2, box1_y2 = box1
        box2_x1, box2_y1, box2_x2, box2_y2 = box2
        x1 = max(box1_x1, box2_x1)
        y1 = max(box1_y1, box2_y1)
        x2 = min(box1_x2, box2_x2)
        y2 = min(box1_y2, box2_y2)
        intersection = (x2 - x1) * (y2 - y1)
        return intersection

    def __union(self, box1: torch.tensor, box2: torch.tensor):
        """
        The method computes the union of two bounding boxes.

        Args:
            box1 (torch.tensor[float]): First bounding box.
            box2 (torch.tensor[float]): Second bounding box.

        Returns:
            union (float): The amount of union between two bounding boxes.
        """
        box1_x1, box1_y1, box1_x2, box1_y2 = box1
        box2_x1, box2_y1, box2_x2, box2_y2 = box2
        box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
        box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
        union = box1_area + box2_area - self.__intersection(box1, box2)
        return union

    def __iou(self, box1: torch.tensor, box2: torch.tensor):
        """
        The method computes the intersection over union of two bounding boxes.

        Args:
            box1 (torch.tensor[float]): First bounding box.
            box2 (torch.tensor[float]): Second bounding box.

        Returns:
            iou (float): Intersection over union for two bounding boxes.
        """
        iou = self.__intersection(box1, box2) / self.__union(box1, box2)
        return iou

    def __non_maximum_supression(self, boxes: list, labels: list, iou: float = 0.5):
        """
        The method implements the non_max_suppression algorithm, which combines several bounding boxes into one,
        depending on the given threshold value iou.

        Args:
            boxes (list[torch.tensor[float]]): List of bounding boxes to merge.
            labels (list[int]): List of classes corresponding to the bounding boxes to be merged.
            iou (float): The boundary value of iou, with which the calculated value between two iou frames is compared.
                         If the calculated value is greater than the specified value, then the two frames are combined
                         into one. In this way, all the bounding frames are searched in pairs.

        Returns:
            nms_boxes (list[torch.tensor[float]]): New list of combined bounding boxes.
            nms_labels (list[int]): New list of classes corresponding to the combined bounding boxes.
        """
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

    def xywh_to_x1y1x2y2(self, boxes: list):
        """
        This method allows you to change the format of the coordinates of the bounding boxes
        from "x, y, w, h" to "x_min, y_min, x_max, y_max".

        Args:
            boxes (list[torch.tensor[float]]): List of bounding box coordinates in "x, y, w, h" format.

        Returns:
            x1y1x2y2_boxes (list[torch.tensor[float]]): List of bounding box coordinates in
                                                        "x_min, y_min, x_max, y_max" format.
        """
        x1y1x2y2_boxes = []
        for box in boxes:
            x1 = float(box[0] - box[2] / 2)
            y1 = float(box[1] + box[3] / 2)
            x2 = float(box[0] + box[2] / 2)
            y2 = float(box[1] - box[3] / 2)
            new_box = torch.tensor([x1 if x1 < x2 else x2,
                                    y1 if y1 < y2 else y2,
                                    x2 if x2 >= x1 else x1,
                                    y2 if y2 >= y1 else y1,
                                    ])
            x1y1x2y2_boxes.append(new_box)
        return x1y1x2y2_boxes

    def x1y1x2y2_to_xywh(self, boxes: list):
        """
        This method allows you to change the format of the coordinates of the bounding boxes
        from "x_min, y_min, x_max, y_max" to "x, y, w, h".

        Args:
            boxes (list[torch.tensor[float]]): List of bounding box coordinates in "x_min, y_min, x_max, y_max" format.

        Returns:
            xywh_boxes (list[torch.tensor[float]]): List of bounding box coordinates in
                                                        "x, y, w, h" format.
        """
        xywh_boxes = []
        for box in boxes:
            x_center = float(box[0] + box[2]) / 2
            y_center = float(box[1] + box[3]) / 2
            w = float(box[2] - box[0])
            h = float(box[3] - box[1])
            xywh_boxes.append(torch.tensor([x_center, y_center, w, h]))
        return xywh_boxes

    def predict(self, image, conf: float = 0.5, iou: float = 0.5, coords_format: str = 'xywh'):
        """
        The method returns the model's prediction (detects a gesture in an image) for a single image.

        Args:
            image (np.array): Image in the form of a numpy-array.
            conf (float): The threshold value of the model's confidence that an object is present in the image.
            iou (float): The threshold value of the intersection over union to combine bounding boxes.
            coords_format (str): The format of the coordinates of the returned bounding boxes.

        Returns:
            result (dict[list, list]): A dictionary containing the bounding boxes of all objects found in the image,
                                       as well as the corresponding object class names.
        """
        if self.model_type == 'YOLO':
            y_pred = self.model.predict(image, verbose=False, show=False, conf=conf, iou=iou)[0]
            if not y_pred.boxes:
                return None
            if coords_format == 'xywh':
                result = {'boxes': y_pred.boxes.xywh, 'labels': y_pred.boxes.cls}
            else:
                result = {'boxes': y_pred.boxes.xyxy, 'labels': y_pred.boxes.cls}
        if self.model_type == 'SSDLite':
            image_tensor = to_tensor(image).unsqueeze(dim=0)
            y_pred = self.model(image_tensor)[0]
            y_pred['boxes'] = y_pred['boxes'][y_pred['scores'] > conf]
            y_pred['labels'] = y_pred['labels'][y_pred['scores'] > conf]
            if not y_pred['boxes'].size()[0]:
                return None
            if coords_format == 'xywh':
                y_pred['boxes'] = self.x1y1x2y2_to_xywh(y_pred['boxes'])
            nms_boxes, nms_labels = self.__non_maximum_supression(y_pred['boxes'], y_pred['labels'], iou)
            result = {'boxes': nms_boxes, 'labels': nms_labels}
        result['labels'] = [NUM_TO_GESTURE[int(cls_name)] for cls_name in result['labels']]
        return result

    def __SSDLite_get_params(self, freeze: int = 0):
        """
        The method returns the parameters of the model, freezing some of them,
        starting from the beginning, if the freeze argument is specified.

        Args:
            freeze (int): The number of parameters that will not change during training.

        Returns:
            params (list[obj]): List of model parameters.
        """
        params = [p for p in self.model.parameters() if p.requires_grad]
        for i, p in enumerate(params):
            if i < freeze:
                p.requires_grad = False
        return params

    def __SSDLite_fit_preparation(self, dataset: object, freeze: int = 219):
        """
        The method prepares options for training the SDDLite model (creates objects necessary for training).

        Args:
            dataset (obj): A dataset object used to load data during training.
            freeze (int): The number of parameters that will not change during training.

        Returns:
            dataloader (obj), optimizer (obj), lr_sheduler (obj): Objects used during training.
        """
        params = self.__SSDLite_get_params(freeze=freeze)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            collate_fn=utils.collate_fn
        )
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
        return dataloader, optimizer, lr_sheduler

    def fit(self, path_to_data: str, epochs: int, final_checkpoint_name: str):
        """
        The method implements training of the model on the specified data set for the specified number of epochs,
        and also saves the resulting model.

        Args:
            path_to_data (str): Path to the data on which the model is trained.
            epochs (int): The number of epochs during which the model is trained.
            final_checkpoint_name (str): The path along which the trained model's checkpoint is saved.

        Returns:
            None.
        """
        if self.model_type == 'YOLO':
            self.model.train(data=os.path.join(path_to_data, 'data.yaml'),
                             epochs=epochs,
                             imgsz=640,
                             freeze=18,
                             project=os.getcwd()
                             )
            self.model.save(final_checkpoint_name)
        if self.model_type == 'SSDLite':
            device = torch.device('cpu')
            self.model.to(device)
            dataset = SSDLiteDataset(path_to_data, self.xywh_to_x1y1x2y2)
            dataloader, optimizer, lr_sheduler = self.__SSDLite_fit_preparation(dataset, freeze=190)
            for epoch in range(epochs):
                train_one_epoch(self.model, optimizer, dataloader, device, epoch, print_freq=10)
                lr_sheduler.step()
            checkpoint = {
                'epoch': epochs,
                'MODEL_STATE': self.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(checkpoint, final_checkpoint_name)
