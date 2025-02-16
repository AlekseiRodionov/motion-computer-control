import os
import sys
import json

from PyQt5.QtCore import QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5 import QtWidgets
from pygrabber.dshow_graph import FilterGraph
import numpy as np
import cv2

from CommandExecutor import CommandExecutor
from interface import Ui_MainWindow


ALL_GESTURES = [
    'grabbing',
    'grip',
    'holy',
    'point',
    'call',
    'three3',
    'timeout',
    'xsign',
    'hand_heart',
    'hand_heart2',
    'little_finger',
    'middle_finger',
    'take_picture',
    'dislike',
    'fist',
    'four',
    'like',
    'mute',
    'ok',
    'one',
    'palm',
    'peace',
    'peace_inverted',
    'rock',
    'stop',
    'stop_inverted',
    'three',
    'three2',
    'two_up',
    'two_up_inverted',
    'three_gun',
    'thumb_index',
    'thumb_index2',
    'no_gesture'
]


class CameraThread(QThread):
    """
    The class implements access to a parallel video stream and is needed
    to display this video stream in the interface of the main application.

    Attributes:
        capture (cv2.VideoCapture): Video capture device.
        is_active (bool): Flag indicating whether the camera is currently active
                          (whether it should be used in the PyQT interface).
    """
    image = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self.capture = None
        self.is_active = False

    def start(self, camera_index: int):
        """
        The method turns on the camera (uses the camera connected to the computer with the number camera_index)
        and enables the transmission of the video stream.

        Args:
            camera_index (int): The number of the camera used.

        Returns:
            None.
        """
        self.is_active = True
        self.capture = cv2.VideoCapture(camera_index)
        super().start()

    def run(self):
        """
        Performs recording and transmission of video stream in parallel with the main application.

        Returns:
            None.
        """
        while self.is_active:
            try:
                ret, frame = self.capture.read()
                frame = cv2.flip(frame, 1)
                if ret:
                    self.image.emit(frame)
            except:
                continue

    def stop(self):
        """
        Stops recording and transmitting the video stream to the main application.

        Returns:
            None.
        """
        self.is_active = False
        if self.capture:
            self.capture.release()
            self.capture = None


class MotionControlApp(QtWidgets.QMainWindow):
    """
    A class implementing the main application interface. Its capabilities are described in detail in README.md.

    Most of the attributes set in __init__ are needed for use on the "Data" tab. To understand what is discussed
    below, it is recommended to first read the README.md of the project
    (the information regarding the "Data" tab).
    Attributes:
        ui (object): The application interface, contained in the interface.py file, was created using the PyQT designer.
        ==================================
        Attributes used in the "Main" tab.
        ==================================
        command_executor (object): An object used to associate gestures with specific Python scripts,
                                   as well as to store sets of commands, and to remove individual commands
                                   from commands files.
        ==================================
        Attributes used in the Data tab.
        ==================================
        camera (object): An object that provides parallel transmission of a video stream.
        camera_timer (object): A timer counting down the time until the photo is taken.
        original_image (object): An image received from a camera, displayed in the interface of the main program,
                                 but not preserving all the transformations performed on it
                                 (all the rectangular boxes drawn on it, etc.).
        current_image (object): An image received from a camera, displayed in the interface of the main program
                                and preserving all the transformations performed on it
                                (all the rectangular boxes drawn on it, etc.).
        bounding_box_coords (list): A list containing the coordinates of the current bounding box
                                    selected by the user on the image.
        current_image_targets (list): A list containing the bounding boxes and class names of objects in the image.
        timer_counter (int): A timer counter used to take photos from a camera with a time delay.
        is_photo_saving_mode (bool): A flag indicating whether a photo has been taken or whether
                                     there is a continuous video stream.
        COLORS (dict): A dictionary containing RGB color codes (in the form of tuples of three elements of type int).
        BOX_TEXT_X_BIAS (int): Offset from the corner of the bounding box along the x-axis.
        BOX_TEXT_Y_BIAS (int): Offset from the corner of the bounding box along the y-axis.
        BOX_THICKNESS (int): Thickness of the bounding box.
        TEXT_THICKNESS (int): Thickness of the bounding box text.
        FONT_SCALE (float): The font scale of the bounding box text.
        POINT_RADIUS (int): The radius of the bounding box point.
        ==================================
        Attributes used in the Train tab.
        ==================================
        None.
    """

    def __init__(self):
        super(MotionControlApp, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.command_executor = CommandExecutor()

        self.camera = CameraThread()
        self.camera_timer = QTimer()
        self.original_image = None
        self.current_image = None
        self.bounding_box_coords = []
        self.current_image_targets = []
        self.timer_counter = 0
        self.is_photo_saving_mode = False
        self.COLORS = {
            'red': (255, 0, 0),
            'orange': (255, 128, 0),
            'yellow': (255, 255, 0),
            'green': (0, 255, 0),
            'light_blue': (0, 255, 255),
            'blue': (0, 0, 255),
            'violet': (128, 0, 255),
            'pink': (255, 0, 255),
            'grey': (128, 128, 128),
            'black': (0, 0, 0),
            'white': (255, 255, 255)
        }
        self.BOX_TEXT_X_BIAS = 4
        self.BOX_TEXT_Y_BIAS = 20
        self.BOX_THICKNESS = 4
        self.TEXT_THICKNESS = 2
        self.FONT_SCALE = 0.8
        self.POINT_RADIUS = 3

        self.ui.cancel_photo_button.setVisible(False)
        self.__get_available_cameras()
        self.__set_black_screen_cam()
        self.__connections()

    def on_create_commands_file_click(self):
        """
        Creates a commands file (in json format) at the specified path.

        Returns:
            None.
        """
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Создать файл', '', 'JSON Files (*.json)')
        if filename:
            self.ui.commands_file_edit.setText(filename)
            with open(filename, 'w', encoding='utf8') as json_file:
                json_file.write('{}')
                json_file.close()

    def on_open_commands_file_click(self):
        """
        Receives from the user via FileDialog the path to the commands file
        and inserts this path into the corresponding widget.

        Returns:
            None.
        """
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Открыть файл', '', 'JSON Files (*.json)')
        if filename:
            self.ui.commands_file_edit.setText(filename)
            self.command_executor.load_commands_dict(filename)

    def on_open_script_click(self):
        """
        Receives from the user via FileDialog the path to the python-script
        and inserts this path into the corresponding widget.

        Returns:
            None.
        """
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Открыть файл', '', 'PYTHON Files (*.py)')
        if filename:
            self.ui.script_path_edit.setText(filename)

    def on_save_command_click(self):
        """
        Saves a command (gesture and its corresponding python script) to the specified command file.

        Returns:
            None.
        """
        gesture = self.ui.gesture_box.currentText()
        script_path = self.ui.script_path_edit.text()
        commands_file_path = self.ui.commands_file_edit.text()
        if os.path.isfile(script_path) and script_path.endswith('.py'):
            if os.path.isfile(commands_file_path) and commands_file_path.endswith('.json'):
                self.command_executor.load_commands_dict(commands_file_path)
                self.command_executor.create_command(commands_file_path, gesture, script_path)
                self.__show_message("Добавление команды",
                                    "Команда успешно добавлена!",
                                    QtWidgets.QMessageBox.Information)
            else:
                message_text = "Файла с командами по указанному пути не существует, либо он не является json-файлом. " \
                               "Проверьте правильность введённого пути."
                self.__show_message("Добавление команды",
                                    message_text,
                                    QtWidgets.QMessageBox.Critical)
        else:
            message_text = "Python-скрипта по указанному пути не существует. " \
                           "Проверьте правильность введённого пути."
            self.__show_message("Добавление команды",
                                message_text,
                                QtWidgets.QMessageBox.Critical)

    def on_delete_command_click(self):
        """
        Removes a command from the specified command file.

        Returns:
            None.
        """
        commands_file_path = self.ui.commands_file_edit.text()
        gesture = self.ui.gesture_box.currentText()
        if os.path.isfile(commands_file_path) and commands_file_path.endswith('.json'):
            self.command_executor.delete_command(commands_file_path, gesture)
            self.__show_message("Удаление команды",
                                "Команда успешно удалена!",
                                QtWidgets.QMessageBox.Information)
        else:
            message_text = "Файла с командами по указанному пути не существует, либо он не является json-файлом. " \
                           "Проверьте правильность введённого пути."
            self.__show_message("Удаление команды",
                                message_text,
                                QtWidgets.QMessageBox.Critical)

    def on_show_commands_click(self):
        """
        Shows which commands are written in the command file (which scripts correspond to which gestures).

        Returns:
            None.
        """
        commands_file_path = self.ui.commands_file_edit.text()
        window_text = ''
        if os.path.isfile(commands_file_path) and commands_file_path.endswith('.json'):
            self.command_executor.load_commands_dict(commands_file_path)
            for gesture, script_path in self.command_executor.commands_dict.items():
                window_text += gesture + ': ' + script_path + '\n\n'
            if window_text:
                self.__show_message("Информация о командах",
                                    window_text,
                                    QtWidgets.QMessageBox.Information)
            else:
                message_text = "В указанном файле не сохранено ни одной команды."
                self.__show_message("Информация о командах",
                                    message_text,
                                    QtWidgets.QMessageBox.Information)
        else:
            message_text = "Файла с командами по указанному пути не существует, либо он не является json-файлом. " \
                           "Проверьте правильность введённого пути."
            self.__show_message("Информация о командах",
                                message_text,
                                QtWidgets.QMessageBox.Critical)

    def on_open_main_checkpoint_click(self):
        """
        Receives from the user via FileDialog the path to the start checkpoint
        of the trained model and inserts this path into the corresponding widget.

        Returns:
            None.
        """
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Открыть файл', '', 'PYTORCH Files (*.pt *.pth)')
        if filename:
            self.ui.main_chkpt_file_edit.setText(filename)

    def on_start_app_click(self):
        """
        Checks the correctness of the entered data, then saves the config file
        for start app and runs application.

        Returns:
            None.
        """
        commands_file_path = self.ui.commands_file_edit.text()
        model_type = self.ui.main_model_type_box.currentText()
        path_to_checkpoint = self.ui.main_chkpt_file_edit.text()
        camera_index = self.ui.main_camera_box.currentIndex()
        confidence = self.ui.main_confidence_edit.text()
        iou = self.ui.main_iou_edit.text()
        if not (os.path.isfile(commands_file_path) and commands_file_path.endswith('.json')):
            message_text = "Файла с командами по указанному пути не существует, либо он не является json-файлом. " \
                           "Проверьте правильность введённого пути."
            self.__show_message("Ошибка в пути к командному файлу",
                                message_text,
                                QtWidgets.QMessageBox.Critical)
        elif not (os.path.isfile(path_to_checkpoint) and
                 (path_to_checkpoint.endswith('.pt') or path_to_checkpoint.endswith('.pth'))):
            message_text = "Чекпойнта по указанному пути не существует, либо он не является файлом с " \
                           "расширением '.pt' или '.pth'. Проверьте правильность указанного пути."
            self.__show_message("Ошибка в пути к чекпойнту",
                                message_text,
                                QtWidgets.QMessageBox.Critical)
        elif model_type == 'YOLO' and not path_to_checkpoint.endswith('.pt'):
            message_text = "Для моделей типа 'YOLO' используются чекпойнты с расширением '.pt'. " \
                           "Проверьте правильность указанного пути к чекпойнту."
            self.__show_message("Ошибка в пути к чекпойнту",
                                message_text,
                                QtWidgets.QMessageBox.Critical)
        elif model_type == 'SSDLite' and not path_to_checkpoint.endswith('.pth'):
            message_text = "Для моделей типа 'SSDLite' используются чекпойнты с расширением '.pth'. " \
                           "Проверьте правильность указанного пути к чекпойнту."
            self.__show_message("Ошибка в пути к чекпойнту",
                                message_text,
                                QtWidgets.QMessageBox.Critical)
        elif not self.__validate_conf_and_iou(confidence):
            message_text = "Значение confidence должно задаваться в пределах от 0 до 1."
            self.__show_message("Ошибка в значении confidence",
                                message_text,
                                QtWidgets.QMessageBox.Critical)
        elif not self.__validate_conf_and_iou(iou):
            message_text = "Значение iou должно задаваться в пределах от 0 до 1."
            self.__show_message("Ошибка в значении iou",
                                message_text,
                                QtWidgets.QMessageBox.Critical)
        else:
            config_dict = {
                'command_file_path': commands_file_path,
                'model_type': model_type,
                'path_to_checkpoint': path_to_checkpoint,
                'camera_index': camera_index,
                'conf': confidence,
                'iou': iou,
            }
            with open(os.path.join('configs', 'start_app_config.json'), 'w', encoding='utf8') as command_file:
                json.dump(config_dict, command_file)
            os.startfile('start_motion_control.py')

    def on_open_data_save_click(self):
        """
        Receives the path to the saving data from the user via FileDialog
        and inserts this path into the corresponding widget.

        Returns:
            None.
        """
        dirname = QtWidgets.QFileDialog.getExistingDirectory(self, 'Выберите папку, куда будут сохраняться данные')
        if dirname:
            self.ui.data_save_folder_edit.setText(dirname)

    def on_use_camera_click(self):
        """
        Turns the camera on/off.

        Returns:
            None.
        """
        if self.ui.use_camera_button.text() == 'Включить камеру':
            self.ui.use_camera_button.setDisabled(True)
            self.ui.use_camera_button.setText('Выключить камеру')
            self.camera.start(self.ui.data_camera_box.currentIndex())
            self.ui.use_camera_button.setEnabled(True)
            self.ui.get_photo_button.setEnabled(True)
        else:
            self.ui.use_camera_button.setDisabled(True)
            self.ui.use_camera_button.setText('Включить камеру')
            self.camera.stop()
            self.ui.use_camera_button.setEnabled(True)
            self.ui.get_photo_button.setDisabled(True)
            self.__set_black_screen_cam()

    def on_get_photo_click(self):
        """
        Takes a photo (stops the video stream on the last frame) and enables the ability to select objects in the image.
        If the photo has already been taken, then saves the data to the training sample instead.

        Returns:
            None.
        """
        data_path = self.ui.data_save_folder_edit.text()
        if not os.path.exists(data_path):
            message_text = "Папки для сохранения данных по указанному пути не существует. " \
                           "Проверьте введённый путь к папке с данными."
            self.__show_message("Ошибка в пути к данным",
                                message_text,
                                QtWidgets.QMessageBox.Critical)
        else:
            if not self.is_photo_saving_mode:
                self.current_image_targets = []
                self.current_image = self.ui.data_image_label.pixmap().toImage()
                self.ui.get_photo_button.setText('Сохранить')
                self.__disable_all_data_widgets()
                self.ui.cancel_photo_button.setVisible(True)
                self.ui.get_photo_button.setDisabled(True)
                self.timer_counter = int(self.ui.timer_edit.text())
                if self.timer_counter > 0:
                    self.camera_timer.start(1000)
                else:
                    self.__get_photo()
            else:
                self.ui.get_photo_button.setText('Сделать фото')
                self.camera.start(self.ui.data_camera_box.currentIndex())
                self.__enable_all_data_widgets()
                self.ui.cancel_photo_button.setDisabled(True)
                self.ui.cancel_photo_button.setVisible(False)
                if self.current_image_targets:
                    self.__save_train_data()
            self.is_photo_saving_mode = not self.is_photo_saving_mode

    def on_cancel_photo_click(self):
        """
        Clears all bounding box selections and resets the current image. No data is saved.

        Returns:
            None.
        """
        self.ui.cancel_photo_button.setDisabled(True)
        self.ui.cancel_photo_button.setVisible(False)
        self.ui.get_photo_button.setText('Сделать фото')
        self.camera.start(self.ui.data_camera_box.currentIndex())
        self.__enable_all_data_widgets()
        self.ui.cancel_photo_button.setDisabled(True)
        self.ui.cancel_photo_button.setVisible(False)
        self.is_photo_saving_mode = not self.is_photo_saving_mode
        self.bounding_box_coords = []

    def on_photo_click(self, event):
        """
        Gets the coordinates of the pixel the user clicked on the image,
        then creates a bounding box based on those coordinates.

        Args:
            event (object): Event, mouse click on the image.

        Returns:
            None.
        """
        if self.is_photo_saving_mode:
            x_coord = event.pos().x()
            y_coord = event.pos().y()
            self.bounding_box_coords.append(x_coord)
            self.bounding_box_coords.append(y_coord)
            self.__object_labeling()

    def on_open_train_folder_click(self):
        """
        Receives the path to the training data from the user via FileDialog
        and inserts this path into the corresponding widget.

        Returns:
            None.
        """
        dirname = QtWidgets.QFileDialog.getExistingDirectory(self, 'Выберите папку с тренировочными данными')
        if dirname:
            self.ui.data_train_edit.setText(dirname)

    def on_open_start_checkpoint_click(self):
        """
        Receives from the user via FileDialog the path to the start checkpoint
        of the trained model and inserts this path into the corresponding widget.

        Returns:
            None.
        """
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Открыть файл', '', 'PYTORCH Files (*.pt *.pth)')
        if filename:
            self.ui.start_chkpt_file_edit.setText(filename)

    def on_open_end_checkpoint_click(self):
        """
        Receives from the user via FileDialog the path to the final checkpoint (where the trained model will be saved)
        of the trained model and inserts this path into the corresponding widget.

        Returns:
            None.
        """
        model_type = self.ui.train_model_type_box.currentText()
        filename = ''
        if model_type == 'YOLO':
            filename, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Укажите путь к итоговому чекпойнту', '',
                                                                'PYTORCH Files (*.pt)')
        elif model_type == 'SSDLite':
            filename, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Укажите путь к итоговому чекпойнту', '',
                                                                'PYTORCH Files (*.pth)')
        if filename:
            self.ui.end_chkpt_path_edit.setText(filename)

    def on_start_training_click(self):
        """
         Checks the correctness of the entered data, then saves the config file
        for training the model and runs the file for training.

        Returns:
            None.
        """
        model_type = self.ui.train_model_type_box.currentText()
        path_to_init_checkpoint = self.ui.start_chkpt_file_edit.text()
        path_to_data = self.ui.data_train_edit.text()
        epochs = self.ui.epochs_edit.text()
        path_to_final_checkpoint = self.ui.end_chkpt_path_edit.text()
        if not os.path.exists(path_to_data):
            message_text = "Папки с тренировочными данными по указанному пути не существует. " \
                           "Проверьте путь к папке с данными."
            self.__show_message("Ошибка в пути к данным",
                                message_text,
                                QtWidgets.QMessageBox.Critical)
        elif not (os.path.exists(os.path.join(path_to_data, 'images')) and
                 (os.path.exists(os.path.join(path_to_data, 'labels'))) and
                 (os.path.exists(os.path.join(path_to_data, 'data.yaml')))):
            message_text = "Папка с тренировочными данными должна содержать подпапки " \
                           "'images' и 'labels', а также файл data.yaml."
            self.__show_message("Ошибка в пути к данным",
                                message_text,
                                QtWidgets.QMessageBox.Critical)
        elif not (os.path.isfile(path_to_init_checkpoint) and
                 (path_to_init_checkpoint.endswith('.pt') or path_to_init_checkpoint.endswith('.pth'))):
            message_text = "Стартового чекпойнта по указанному пути не существует, либо он не является " \
                           "файлом с расширением '.pt' или '.pth'. Проверьте правильность указанного пути."
            self.__show_message("Ошибка в пути к стартовому чекпойнту",
                                message_text,
                                QtWidgets.QMessageBox.Critical)
        elif not ((path_to_init_checkpoint.endswith('.pt') and path_to_final_checkpoint.endswith('.pt')) or
                  (path_to_init_checkpoint.endswith('.pth') and path_to_final_checkpoint.endswith('.pth'))):
            message_text = "Расширение стартового чекпойнта и расширение конечного чекпойнта должны совпадать."
            self.__show_message("Несоответствие чекпойнтов",
                                message_text,
                                QtWidgets.QMessageBox.Critical)
        elif model_type == 'YOLO' and not (path_to_init_checkpoint.endswith('.pt') or
                                           path_to_final_checkpoint.endswith('.pt')):
            message_text = "Для моделей типа 'YOLO' используются чекпойнты с расширением '.pt'. " \
                           "Проверьте правильность указанного пути к стартовому и конечному чекпойнтам."
            self.__show_message("Ошибка в пути к стартовому чекпойнту",
                                message_text,
                                QtWidgets.QMessageBox.Critical)
        elif model_type == 'SSDLite' and not (path_to_init_checkpoint.endswith('.pth') or
                                              path_to_final_checkpoint.endswith('.pth')):
            message_text = "Для моделей типа 'SSDLite' используются чекпойнты с расширением '.pth'. " \
                           "Проверьте правильность указанного пути к стартовому и конечному чекпойнтам."
            self.__show_message("Ошибка в пути к стартовому чекпойнту",
                                message_text,
                                QtWidgets.QMessageBox.Critical)
        else:
            config_dict = {
                'model_type': model_type,
                'path_to_init_checkpoint': path_to_init_checkpoint,
                'path_to_data': path_to_data,
                'epochs': epochs,
                'path_to_final_checkpoint': path_to_final_checkpoint
            }
            with open(os.path.join('configs', 'train_config.json'), 'w', encoding='utf8') as command_file:
                json.dump(config_dict, command_file)
            os.startfile('detector_training.py')

    def on_open_test_checkpoint_click(self):
        """
        Receives the path to the checkpoint of the model being tested from the user via FileDialog
        and inserts this path into the corresponding widget.

        Returns:
            None.
        """
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Открыть файл', '', 'PYTORCH Files (*.pt *.pth)')
        if filename:
            self.ui.test_chkpt_path_edit.setText(filename)

    def on_start_test_click(self):
        """
        Checks the correctness of the entered data, then saves the config file
        for testing the model and runs the file for testing.

        Returns:
            None.
        """
        model_type = self.ui.test_model_type_box.currentText()
        path_to_checkpoint = self.ui.test_chkpt_path_edit.text()
        camera_index = self.ui.test_camera_box.currentIndex()
        confidence = self.ui.test_confidence_edit.text()
        iou = self.ui.test_iou_edit.text()
        if not (os.path.isfile(path_to_checkpoint) and
               (path_to_checkpoint.endswith('.pt') or path_to_checkpoint.endswith('.pth'))):
            message_text = "Чекпойнта по указанному пути не существует, либо он не является файлом с расширением " \
                           "'.pt' или '.pth'. Проверьте правильность указанного пути."
            self.__show_message("Ошибка в пути к чекпойнту",
                                message_text,
                                QtWidgets.QMessageBox.Critical)
        elif model_type == 'YOLO' and not path_to_checkpoint.endswith('.pt'):
            message_text = "Для моделей типа 'YOLO' используются чекпойнты с расширением '.pt'. " \
                           "Проверьте правильность указанного пути к чекпойнту."
            self.__show_message("Ошибка в пути к чекпойнту",
                                message_text,
                                QtWidgets.QMessageBox.Critical)
        elif model_type == 'SSDLite' and not path_to_checkpoint.endswith('.pth'):
            message_text = "Для моделей типа 'SSDLite' используются чекпойнты с расширением '.pth'. " \
                           "Проверьте правильность указанного пути к чекпойнту."
            self.__show_message("Ошибка в пути к чекпойнту",
                                message_text,
                                QtWidgets.QMessageBox.Critical)
        elif not self.__validate_conf_and_iou(confidence):
            message_text = "Значение confidence должно задаваться в пределах от 0 до 1."
            self.__show_message("Ошибка в значении confidence",
                                message_text,
                                QtWidgets.QMessageBox.Critical)
        elif not self.__validate_conf_and_iou(iou):
            message_text = "Значение iou должно задаваться в пределах от 0 до 1."
            self.__show_message("Ошибка в значении iou",
                                message_text,
                                QtWidgets.QMessageBox.Critical)
        else:
            config_dict = {
                'model_type': model_type,
                'path_to_checkpoint': path_to_checkpoint,
                'camera_index': camera_index,
                'conf': confidence,
                'iou': iou
            }
            with open(os.path.join('configs', 'test_config.json'), 'w', encoding='utf8') as command_file:
                json.dump(config_dict, command_file)
            os.startfile('detector_evaluation.py')

    def __connections(self):
        """
        Binds the execution of methods to the corresponding events (clicking on a button, clicking on a photo, etc.)

        Returns:
            None.
        """
        self.ui.create_commands_file_button.clicked.connect(self.on_create_commands_file_click)
        self.ui.get_commands_file_button.clicked.connect(self.on_open_commands_file_click)
        self.ui.get_script_file_button.clicked.connect(self.on_open_script_click)
        self.ui.save_command_button.clicked.connect(self.on_save_command_click)
        self.ui.get_main_chkpt_button.clicked.connect(self.on_open_main_checkpoint_click)
        self.ui.start_app_button.clicked.connect(self.on_start_app_click)
        self.ui.get_data_save_folder_button.clicked.connect(self.on_open_data_save_click)
        self.ui.delete_commands_button.clicked.connect(self.on_delete_command_click)
        self.ui.show_commands_button.clicked.connect(self.on_show_commands_click)
        self.ui.use_camera_button.clicked.connect(self.on_use_camera_click)
        self.ui.get_data_train_folder_button.clicked.connect(self.on_open_train_folder_click)
        self.ui.get_start_chkpt_file_button.clicked.connect(self.on_open_start_checkpoint_click)
        self.ui.start_train_button.clicked.connect(self.on_start_training_click)
        self.ui.create_end_chkpt_file_button.clicked.connect(self.on_open_end_checkpoint_click)
        self.ui.start_test_button.clicked.connect(self.on_start_test_click)
        self.ui.get_test_chkpt_file_button.clicked.connect(self.on_open_test_checkpoint_click)
        self.ui.get_photo_button.clicked.connect(self.on_get_photo_click)
        self.ui.cancel_photo_button.clicked.connect(self.on_cancel_photo_click)
        self.camera_timer.timeout.connect(self.__get_photo_by_timer)
        self.camera.image.connect(self.__update_image)
        self.ui.data_image_label.mousePressEvent = self.on_photo_click

    def __get_available_cameras(self):
        """
        Gets a list of available cameras connected to the computer and adds them to the corresponding combo boxes.

        Returns:
            None.
        """
        cameras = FilterGraph().get_input_devices()
        for camera_name in cameras:
            self.ui.main_camera_box.addItem(camera_name)
            self.ui.data_camera_box.addItem(camera_name)
            self.ui.test_camera_box.addItem(camera_name)

    def __show_message(self, window_name: str, text: str, icon: int):
        """
        Shows a message in a MessageBox containing only text and a single ok button.

        Args:
            window_name (str): The title of the pop-up window.
            text (str): The text displayed in the window.
            icon (int): Icon used in the window (information, error, etc.)

        Returns:
            None.
        """
        msg = QtWidgets.QMessageBox()
        msg.setWindowTitle(window_name)
        msg.setIcon(icon)
        msg.setText(text)
        msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
        msg.exec_()

    def __validate_conf_and_iou(self, value: str):
        """
        Checks if the value is a number between 0 and 1 and can be converted to float.

        Args:
            value (str): The confidence or interception value over a union of type str.

        Returns:
            (bool): A flag indicating whether the value can be converted to float and whether it is between 0 and 1.
        """
        try:
            float_value = float(value)
            if not 0.0 <= float_value <= 1.0:
                raise ValueError()
            return True
        except ValueError:
            return False

    def __update_image(self, cv2_image: np.ndarray):
        """
        Receives an image from a camera running in a parallel stream and updates it in the program interface.

        Args:
            cv2_image (np.ndarray): Image as a numpy array.

        Returns:
            None.
        """
        cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        self.original_image = QImage(cv2_image, cv2_image.shape[1], cv2_image.shape[0], QImage.Format_RGB888)
        self.ui.data_image_label.setPixmap(QPixmap.fromImage(self.original_image))

    def __set_black_screen_cam(self):
        """
        Updates the image in the program interface by inserting a black background. Used when the camera is disabled.

        Returns:
            None.
        """
        cv2_image = np.zeros((480, 640))
        self.original_image = QImage(cv2_image, cv2_image.shape[1], cv2_image.shape[0], QImage.Format_RGB888)
        self.ui.data_image_label.setPixmap(QPixmap.fromImage(self.original_image))

    def __get_photo(self):
        """
        Stops the video stream and leaves the last frame in the pixmap.

        Returns:
            None.
        """
        self.camera.stop()
        self.ui.get_photo_button.setEnabled(True)
        self.ui.cancel_photo_button.setEnabled(True)
        self.ui.timer_edit.setText('0')

    def __get_photo_by_timer(self):
        """
        Stops the video stream and leaves the last frame in the pixmap.
        The stream is stopped by a timer - with a delay specified by the user.

        Returns:
            None.
        """
        if self.is_photo_saving_mode:
            self.timer_counter -= 1
            self.ui.timer_edit.setText(f'{self.timer_counter}')
            if self.timer_counter == 0:
                self.__get_photo()
                self.camera_timer.stop()

    def __disable_all_data_widgets(self):
        """
        Disables the widgets in the Data tab, leaving enabled only
        those responsible for saving data or canceling results.

        Returns:
            None.
        """
        self.ui.data_save_folder_edit.setDisabled(True)
        self.ui.get_data_save_folder_button.setDisabled(True)
        self.ui.data_camera_box.setDisabled(True)
        self.ui.use_camera_button.setDisabled(True)
        self.ui.timer_edit.setDisabled(True)

    def __enable_all_data_widgets(self):
        """
        Enables the Data tab widgets.

        Returns:
            None.
        """
        self.ui.data_save_folder_edit.setEnabled(True)
        self.ui.get_data_save_folder_button.setEnabled(True)
        self.ui.data_camera_box.setEnabled(True)
        self.ui.use_camera_button.setEnabled(True)
        self.ui.timer_edit.setEnabled(True)

    def __convert_qt_image_to_cv2_image(self, qt_image: object):
        """
        Converts a QImage to an open-cv image (an array of type numpy-ndarray).

        Args:
            qt_image (object): Image of type QImage.

        Returns:
            cv2_image (np.ndarray): Image of type numpy-ndarray.
        """
        qt_image = qt_image.convertToFormat(4)
        width = qt_image.width()
        height = qt_image.height()
        bits_of_image = qt_image.bits()
        bits_of_image.setsize(qt_image.byteCount())
        cv2_image = np.array(bits_of_image).reshape(height, width, 4)
        return cv2_image

    def __paint_point(self, x: int, y: int):
        """
        Draws a point with coordinates x, y on the pixmap.

        Args:
            x (int): Coordinate x.
            y (int): Coordinate y.

        Returns:
            None.
        """
        qt_image = self.ui.data_image_label.pixmap().toImage()
        cv2_image = self.__convert_qt_image_to_cv2_image(qt_image)
        cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        cv2_new_image = cv2.circle(cv2_image,
                                   (x, y),
                                   radius=self.POINT_RADIUS,
                                   color=self.COLORS['red'],
                                   thickness=-1)
        new_qt_image = QImage(cv2_new_image, cv2_new_image.shape[1], cv2_new_image.shape[0], QImage.Format_RGB888)
        self.ui.data_image_label.setPixmap(QPixmap.fromImage(new_qt_image))

    def __paint_rectangle(self, x1: int, y1: int, x2: int, y2: int):
        """
        Draws a rectangle on a pixel map using two points (one is the lower left corner of the rectangle,
        the other is the upper right) with coordinates x1, y1, x2, y2, respectively.

        Args:
            x1 (int): The x coordinate of the first point.
            y1 (int): The y coordinate of the first point.
            x2 (int): The x coordinate of the second point.
            y2 (int): The y coordinate of the second point.

        Returns:
            None.
        """
        qt_image = self.ui.data_image_label.pixmap().toImage()
        cv2_image = self.__convert_qt_image_to_cv2_image(qt_image)
        cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        cv2_new_image = cv2.rectangle(cv2_image,
                                      (x1, y1),
                                      (x2, y2),
                                      color=self.COLORS['red'],
                                      thickness=self.BOX_THICKNESS)
        new_qt_image = QImage(cv2_new_image, cv2_new_image.shape[1], cv2_new_image.shape[0], QImage.Format_RGB888)
        self.ui.data_image_label.setPixmap(QPixmap.fromImage(new_qt_image))

    def __paint_class_name(self, x: int, y: int, class_name: str):
        """
        Applies text over the image - the name of the class to which the object selected by the bounding box belongs.

        Args:
            x (int): The x coordinate of the beginning of the text.
            y (int): The y coordinate of the beginning of the text.
            class_name (str): Text (class name) written over the image.

        Returns:
            None.
        """
        qt_image = self.ui.data_image_label.pixmap().toImage()
        cv2_image = self.__convert_qt_image_to_cv2_image(qt_image)
        cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        cv2_new_image = cv2.putText(cv2_image, class_name,
                                    (x + self.BOX_TEXT_X_BIAS, y + self.BOX_TEXT_Y_BIAS),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    fontScale=self.FONT_SCALE,
                                    color=self.COLORS['red'],
                                    thickness=self.TEXT_THICKNESS)
        new_qt_image = QImage(cv2_new_image, cv2_new_image.shape[1], cv2_new_image.shape[0], QImage.Format_RGB888)
        self.ui.data_image_label.setPixmap(QPixmap.fromImage(new_qt_image))

    def __save_image(self, path: str, image: object):
        """
        Saves the image to the specified path.

        Args:
            path (str): The path where the image is saved.
            image (object): The saved image is of type QImage.

        Returns:
            image_name (str): The name under which the image was saved (without extension).
        """
        all_images = os.listdir(path)
        image_name = len(all_images)
        image.save(os.path.join(path, f'{image_name}.jpg'))
        return image_name

    def __save_label(self, path: str, image: object, image_name: str, targets: list):
        """
        Stores the coordinates of the bounding boxes of objects (normalized with respect to the image dimensions),
        as well as their corresponding class names.

        Args:
            path (str): The path where the file with information is saved.
            image (object): Image of type QImage.
            image_name (str): The name of the image file. Information about the bounding boxes corresponding
                              to the image is saved under the same name but with the .txt extension.
            targets (list): A list containing the bounding boxes and class names of all objects selected in the image.

        Returns:
            None.
        """
        size = image.size()
        width = size.width()
        height = size.height()
        with open(os.path.join(path, 'labels', f'{image_name}.txt'), 'w') as label_file:
            for target in targets:
                coords = target[1:]
                coords = self.__xyxy_to_xywh(coords)
                x_center, y_center, w, h = self.__normalize_bounding_box_coords(width, height, coords)
                label_file.write(f'{ALL_GESTURES.index(target[0])} {x_center} {y_center} {w} {h}\n')

    def __normalize_bounding_box_coords(self, width: int, height: int, coords: list):
        """
        Normalizes coordinates (scales from 0 to 1) relative to the height and width of the image.

        Args:
            width (int): Image width.
            height (int): Image height.
            coords (list): The coordinates of the bounding box in absolute values.

        Returns:
            coords (list): The coordinates of the bounding box in normalized values.
        """
        coords[0] /= width
        coords[1] /= height
        coords[2] /= width
        coords[3] /= height
        return coords

    def __xyxy_to_xywh(self, coords: list):
        """
        Changes the bounding box coordinate format from (x1, y1, x2, y2) to (x_center, y_center, width, height).

        Args:
            coords (list): List of bounding box coordinates in the format (x1, y1, x2, y2).

        Returns:
            new_coords (list): List of bounding box coordinates in the format (x_center, y_center, width, height).
        """
        x_center = ((coords[0] + coords[2]) / 2)
        y_center = ((coords[1] + coords[3]) / 2)
        w = abs(coords[0] - coords[2])
        h = abs(coords[1] - coords[3])
        new_coords = [x_center, y_center, w, h]
        return new_coords

    def __create_yaml_text(self, path: str):
        """
        Generates the yaml file text needed to configure the training of YOLO models.

        Args:
            path (str): Path to the folder with training data.

        Returns:
            yaml_text (str): Text describing the training settings for YOLO models.
        """
        yaml_text = f'path: {path}\n'
        yaml_text += f'train: images\n'
        yaml_text += f'val: images\n\n'
        yaml_text += f'nc: {len(ALL_GESTURES)}\n\n'
        yaml_text += f'names: {ALL_GESTURES}'
        return yaml_text

    def __save_train_data(self):
        """
        Saves the current photograph as a training sample, as well as the bounding boxes
        and corresponding class names of all objects selected in the image.

        Returns:
            None.
        """
        path = self.ui.data_save_folder_edit.text()
        image = self.original_image
        targets = self.current_image_targets
        if not os.path.exists(os.path.join(path, 'images')):
            os.mkdir(os.path.join(path, 'images'))
        image_name = self.__save_image(os.path.join(path, 'images'), image)
        if not os.path.exists(os.path.join(path, 'labels')):
            os.mkdir(os.path.join(path, 'labels'))
        self.__save_label(path, image, image_name, targets)
        if not os.path.isfile(os.path.join(path, 'data.yaml')):
            yaml_text = self.__create_yaml_text(path)
            with open(os.path.join(path, 'data.yaml'), 'w', encoding='utf8') as yaml_file:
                yaml_file.write(yaml_text)

    def __class_labeling(self):
        """
        Applies the user-entered object class to the image and adds it to the list of the corresponding bounding box.

        Returns:
            None.
        """
        text, is_ok = QtWidgets.QInputDialog.getText(self, 'Название класса', 'Введите название класса: ')
        if is_ok:
            if text not in ALL_GESTURES:
                message_text = "Жеста с указанным названием не существует. Сверьтесь со списком доступных жестов."
                self.__show_message("Ошибка в имени класса",
                                    message_text,
                                    QtWidgets.QMessageBox.Critical)
                self.ui.data_image_label.setPixmap(QPixmap.fromImage(self.current_image))
            else:
                self.__paint_class_name(self.bounding_box_coords[0], self.bounding_box_coords[1], text)
                self.current_image = self.ui.data_image_label.pixmap().toImage()
                self.current_image_targets.append([text] + self.bounding_box_coords)
        else:
            self.ui.data_image_label.setPixmap(QPixmap.fromImage(self.current_image))

    def __object_labeling(self):
        """
        Overlays a point, bounding box, and text entered by the user onto the image.
        Adds the class name and bounding box to the list of labels for further work with them.

        Returns:
            None.
        """
        if len(self.bounding_box_coords) == 2:
            self.__paint_point(self.bounding_box_coords[0], self.bounding_box_coords[1])
        else:
            self.__paint_point(self.bounding_box_coords[2], self.bounding_box_coords[3])
            self.__paint_rectangle(self.bounding_box_coords[0], self.bounding_box_coords[1],
                                   self.bounding_box_coords[2], self.bounding_box_coords[3])
            self.__class_labeling()
            self.bounding_box_coords = []


app = QtWidgets.QApplication([])
application = MotionControlApp()
application.show()
sys.exit(app.exec())
