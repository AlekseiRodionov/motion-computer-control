from PyQt5 import QtWidgets, QtGui
from PyQt5.QtWidgets import QInputDialog

from interface import Ui_MainWindow
import sys
import os
import json
from pygrabber.dshow_graph import FilterGraph
import cv2
import numpy as np

from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor

from GestureDetector import GestureDetector
from CommandExecutor import CommandExecutor

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

class CameraThread(QThread):
    image = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self.capture = None
        self.is_active = False

    def start(self, camera_index):
        self.is_active = True
        self.capture = cv2.VideoCapture(camera_index)
        super().start()

    def run(self):
        while self.is_active:
            try:
                ret, frame = self.capture.read()
                frame = cv2.flip(frame, 1)
                if ret:
                   self.image.emit(frame)
            except:
                continue

    def stop(self):
        self.is_active = False
        if self.capture:
            self.capture.release()
            self.capture = None


class mywindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(mywindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.command_executor = CommandExecutor()
        self.__buttons_connect()
        self.__get_available_cameras()
        self.camera = CameraThread()
        self.__set_black_screen_cam()
        self.is_saving_photo = False
        self.coords_for_saving = []
        self.ui.pushButton_37.setVisible(False)
        self.camera_timer = QTimer()
        self.camera_timer.timeout.connect(self.__get_photo_with_timer)

    def __buttons_connect(self):
        self.ui.pushButton.clicked.connect(self.on_open_json_click)
        self.ui.pushButton_2.clicked.connect(self.on_create_json_click)
        self.ui.pushButton_4.clicked.connect(self.on_open_script_click)
        self.ui.pushButton_5.clicked.connect(self.on_save_command_click)
        self.ui.pushButton_6.clicked.connect(self.on_open_checkpoint_click)
        self.ui.pushButton_7.clicked.connect(self.on_start_click)
        self.ui.pushButton_8.clicked.connect(self.on_open_data_click)
        self.ui.pushButton_19.clicked.connect(self.on_delete_command_click)
        self.ui.pushButton_20.clicked.connect(self.on_print_commands_click)
        self.ui.pushButton_18.clicked.connect(self.on_start_stop_camera_click)
        self.ui.pushButton_11.clicked.connect(self.on_data_for_training_click)
        self.ui.pushButton_12.clicked.connect(self.on_start_checkpoint_for_training_click)
        self.ui.pushButton_13.clicked.connect(self.on_start_training_click)
        self.ui.pushButton_16.clicked.connect(self.on_new_checkpoint_for_training_click)
        self.ui.pushButton_17.clicked.connect(self.on_test_button_click)
        self.ui.pushButton_14.clicked.connect(self.on_test_checkpoint_click)
        self.ui.pushButton_10.clicked.connect(self.on_get_photo_click)
        self.ui.pushButton_37.clicked.connect(self.on_cancel_photo_click)
        self.ui.label_25.mousePressEvent = self.get_x_y

    def __get_available_cameras(self):
        devices = FilterGraph().get_input_devices()
        for device_index, device_name in enumerate(devices):
            self.ui.comboBox_3.addItem(device_name)
            self.ui.comboBox_4.addItem(device_name)
            self.ui.comboBox_13.addItem(device_name)

    def on_create_json_click(self):
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Создать файл', '', 'JSON Files (*.json)')
        if filename:
            self.ui.lineEdit.setText(filename)
            with open(filename, 'w', encoding='utf8') as json_file:
                json_file.write('{}')
                json_file.close()

    def on_open_json_click(self):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Открыть файл', '', 'JSON Files (*.json)')
        if filename:
            self.ui.lineEdit.setText(filename)
            self.command_executor.load_commands_dict(filename)

    def on_open_script_click(self):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Открыть файл', '', 'PYTHON Files (*.py)')
        if filename:
            self.ui.lineEdit_2.setText(filename)

    def on_save_command_click(self):
        gesture = self.ui.comboBox.currentIndex()
        script_path = self.ui.lineEdit_2.text()
        msg = QtWidgets.QMessageBox()
        msg.setWindowTitle("Добавление команды.")
        if os.path.isfile(script_path) and script_path.endswith('.py'):
            self.command_executor.load_commands_dict(self.ui.lineEdit.text())
            self.command_executor.create_command(self.ui.lineEdit.text(), gesture, script_path)
            msg.setIcon(QtWidgets.QMessageBox.Information)
            msg.setText("Команда успешно добавлена!")
        else:
            msg.setIcon(QtWidgets.QMessageBox.Critical)
            msg.setText("Python-скрипта по указанному пути не существует. Проверьте правильность введённого пути.")
        msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
        msg.exec_()

    def on_delete_command_click(self):
        gesture = self.ui.comboBox.currentIndex()
        msg = QtWidgets.QMessageBox()
        msg.setWindowTitle("Удаление команды.")
        self.command_executor.delete_command(self.ui.lineEdit.text(), gesture)
        msg.setIcon(QtWidgets.QMessageBox.Information)
        msg.setText("Команда успешно удалена!")
        msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
        msg.exec_()

    def on_print_commands_click(self):
        msg = QtWidgets.QMessageBox()
        msg.setWindowTitle("Информация о командах: ")
        window_text = ''
        if os.path.exists(self.ui.lineEdit.text()) and self.ui.lineEdit.text().endswith('.json'):
            self.command_executor.load_commands_dict(self.ui.lineEdit.text())
            for gesture_index, script_path in self.command_executor.commands_dict.items():
                window_text += ALL_GESTURES[gesture_index] + ': ' + script_path + '\n\n'
            if window_text:
                msg.setText(window_text)
            else:
                msg.setText('В указанном файле не сохранено ни одной команды.')
        else:
            msg.setText('Файла по указанному пути не существует, либо он не является json-файлом.')
        msg.setIcon(QtWidgets.QMessageBox.Information)
        msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
        msg.exec_()

    def on_open_checkpoint_click(self):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Открыть файл', '')
        if filename:
            self.ui.lineEdit_3.setText(filename)

    def on_start_click(self):
        config_dict = {
            'command_file_path': self.ui.lineEdit.text(),
            'model_type': self.ui.comboBox_2.currentText(),
            'checkpoint_path': self.ui.lineEdit_3.text(),
            'camera_index': self.ui.comboBox_3.currentIndex(),
            'conf': self.ui.lineEdit_4.text(),
            'iou': self.ui.lineEdit_5.text(),
        }
        with open('config.json', 'w', encoding='utf8') as command_file:
            json.dump(config_dict, command_file)
        os.startfile('start_motion_control.py')

    def on_open_data_click(self):
        dirname = QtWidgets.QFileDialog.getExistingDirectory(self, 'Выберите папку, куда будут сохраняться данные')
        if dirname:
            self.ui.lineEdit_6.setText(dirname)

    def on_start_stop_camera_click(self):
        if self.ui.pushButton_18.text() == 'Включить камеру':
            self.ui.pushButton_18.setDisabled(True)
            self.ui.pushButton_18.setText('Выключить камеру')
            self.camera.image.connect(self.__update_image)
            self.camera.start(self.ui.comboBox_4.currentIndex())
            self.ui.pushButton_18.setEnabled(True)
            self.ui.pushButton_10.setEnabled(True)
        else:
            self.ui.pushButton_18.setDisabled(True)
            self.ui.pushButton_18.setText('Включить камеру')
            self.camera.stop()
            self.ui.pushButton_18.setEnabled(True)
            self.ui.pushButton_10.setDisabled(True)
            self.__set_black_screen_cam()

    def __update_image(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.image = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
        self.ui.label_25.setPixmap(QPixmap.fromImage(self.image))

    def __set_black_screen_cam(self):
        frame = np.zeros((480, 640))
        self.image = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
        self.ui.label_25.setPixmap(QPixmap.fromImage(self.image))

    def __get_photo(self):
        self.camera.stop()
        self.ui.pushButton_10.setEnabled(True)
        self.ui.pushButton_37.setEnabled(True)
        self.ui.lineEdit_8.setText('0')

    def __get_photo_with_timer(self):
        if self.is_saving_photo:
            self.timer_duration -= 1
            self.ui.lineEdit_8.setText(f'{self.timer_duration}')
            if self.timer_duration == 0:
                self.__get_photo()
                self.camera_timer.stop()


    def on_get_photo_click(self):
        if not self.is_saving_photo:
            self.coords_list = []
            self.current_image = self.ui.label_25.pixmap().toImage()
            self.ui.pushButton_10.setText('Сохранить')
            self.ui.pushButton_37.setVisible(True)
            self.__disable_all_data_widgets()
            self.ui.pushButton_10.setDisabled(True)
            self.timer_duration = int(self.ui.lineEdit_8.text())
            if self.timer_duration > 0:
                self.camera_timer.start(1000)
            else:
                self.__get_photo()
        else:
            self.ui.pushButton_10.setText('Сделать фото')
            self.camera.start(self.ui.comboBox_4.currentIndex())
            self.__enable_all_data_widgets()
            self.ui.pushButton_37.setDisabled(True)
            self.ui.pushButton_37.setVisible(False)
            if self.coords_list:
                self.__save_train_data(self.ui.lineEdit_6.text(), self.image, self.coords_list)
        self.is_saving_photo = not self.is_saving_photo

    def on_cancel_photo_click(self):
        self.ui.pushButton_37.setDisabled(True)
        self.ui.pushButton_37.setVisible(False)
        self.ui.pushButton_10.setText('Сделать фото')
        self.camera.start(self.ui.comboBox_4.currentIndex())
        self.__enable_all_data_widgets()
        self.ui.pushButton_37.setDisabled(True)
        self.ui.pushButton_37.setVisible(False)
        self.is_saving_photo = not self.is_saving_photo

    def __disable_all_data_widgets(self):
        self.ui.lineEdit_6.setDisabled(True)
        self.ui.pushButton_8.setDisabled(True)
        self.ui.comboBox_4.setDisabled(True)
        self.ui.pushButton_18.setDisabled(True)
        self.ui.lineEdit_8.setDisabled(True)

    def __enable_all_data_widgets(self):
        self.ui.lineEdit_6.setEnabled(True)
        self.ui.pushButton_8.setEnabled(True)
        self.ui.comboBox_4.setEnabled(True)
        self.ui.pushButton_18.setEnabled(True)
        self.ui.lineEdit_8.setEnabled(True)

    def __convertQImageToCV2Image(self, incomingImage):
        incomingImage = incomingImage.convertToFormat(4)
        width = incomingImage.width()
        height = incomingImage.height()
        bits_of_image = incomingImage.bits()
        bits_of_image.setsize(incomingImage.byteCount())
        arr = np.array(bits_of_image).reshape(height, width, 4)  # Copies the data
        return arr

    def __paint_point(self, x, y):
        qimage = self.ui.label_25.pixmap().toImage()
        cv2_current_image = self.__convertQImageToCV2Image(qimage)
        cv2_current_image = cv2.cvtColor(cv2_current_image, cv2.COLOR_BGR2RGB)
        cv2_new_image = cv2.circle(cv2_current_image, (x, y), radius=3, color=(255, 0, 0), thickness=-1)
        new_qimage = QImage(cv2_new_image, cv2_new_image.shape[1], cv2_new_image.shape[0], QImage.Format_RGB888)
        self.ui.label_25.setPixmap(QPixmap.fromImage(new_qimage))

    def __paint_rectangle(self, x1, y1, x2, y2):
        qimage = self.ui.label_25.pixmap().toImage()
        cv2_current_image = self.__convertQImageToCV2Image(qimage)
        cv2_current_image = cv2.cvtColor(cv2_current_image, cv2.COLOR_BGR2RGB)
        cv2_new_image = cv2.rectangle(cv2_current_image, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=4)
        new_qimage = QImage(cv2_new_image, cv2_new_image.shape[1], cv2_new_image.shape[0], QImage.Format_RGB888)
        self.ui.label_25.setPixmap(QPixmap.fromImage(new_qimage))

    def __paint_class_name(self, x_center, y_center, text):
        qimage = self.ui.label_25.pixmap().toImage()
        cv2_current_image = self.__convertQImageToCV2Image(qimage)
        cv2_current_image = cv2.cvtColor(cv2_current_image, cv2.COLOR_BGR2RGB)
        cv2_new_image = cv2.putText(cv2_current_image, text, (x_center + 3, y_center + 18), cv2.FONT_HERSHEY_SIMPLEX,
                                    fontScale=0.8, color=(255, 0, 0), thickness=2)
        new_qimage = QImage(cv2_new_image, cv2_new_image.shape[1], cv2_new_image.shape[0], QImage.Format_RGB888)
        self.ui.label_25.setPixmap(QPixmap.fromImage(new_qimage))

    def __save_train_data(self, path, image, coords_list):
        if not os.path.exists(os.path.join(path, 'images')):
            os.mkdir(os.path.join(path, 'images'))
        all_images = os.listdir(os.path.join(path, 'images'))
        image_name = len(all_images)
        image.save(os.path.join(path, 'images', f'{image_name}.jpg'))
        if not os.path.exists(os.path.join(path, 'labels')):
            os.mkdir(os.path.join(path, 'labels'))
        size = image.size()
        width = size.width()
        height = size.height()
        with open(os.path.join(path, 'labels', f'{image_name}.txt'), 'w') as label_file:
            for coords in coords_list:
                x_center = ((coords[1] + coords[3]) / 2) / width
                y_center = ((coords[2] + coords[4]) / 2) / height
                w = abs(coords[1] - coords[3]) / width
                h = abs(coords[2] - coords[4]) / height
                label_file.write(f'{list(ALL_GESTURES.values()).index(coords[0])} {x_center} {y_center} {w} {h}\n')
        if not os.path.isfile(os.path.join(path, 'data.yaml')):
            yaml_text = f'path: {path}\n'
            yaml_text += f'train: images\n'
            yaml_text += f'val: images\n\n'
            yaml_text += f'nc: {len(ALL_GESTURES.values())}\n\n'
            yaml_text += f'names: {list(ALL_GESTURES.values())}'
            with open(os.path.join(path, 'data.yaml'), 'w', encoding='utf8') as yaml_file:
                yaml_file.write(yaml_text)

    def get_x_y(self, event):
        if self.is_saving_photo:
            x_coord = event.pos().x()
            y_coord = event.pos().y()
            self.coords_for_saving.append(x_coord)
            self.coords_for_saving.append(y_coord)
            if len(self.coords_for_saving) < 3:
                self.__paint_point(x_coord, y_coord)
            else:
                self.__paint_rectangle(self.coords_for_saving[0], self.coords_for_saving[1],
                                       self.coords_for_saving[2], self.coords_for_saving[3])
                text, ok = QInputDialog.getText(self, 'Название класса', 'Введите название класса: ')
                if ok:
                    self.__paint_class_name(self.coords_for_saving[0], self.coords_for_saving[1], text)
                    self.current_image = self.ui.label_25.pixmap().toImage()
                    self.coords_list.append([text] + self.coords_for_saving)
                else:
                    self.ui.label_25.setPixmap(QPixmap.fromImage(self.current_image))
                self.coords_for_saving = []

    def on_data_for_training_click(self):
        dirname = QtWidgets.QFileDialog.getExistingDirectory(self, 'Выберите папку, в которой хранятся тренировочные данные')
        if dirname:
            self.ui.lineEdit_10.setText(dirname)

    def on_start_checkpoint_for_training_click(self):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Открыть файл', '')
        if filename:
            self.ui.lineEdit_9.setText(filename)

    def on_new_checkpoint_for_training_click(self):
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Укажите путь к итоговому чекпойнту', '')
        if filename:
            self.ui.lineEdit_16.setText(filename)

    def on_start_training_click(self):
        config_dict = {
            'model_type': self.ui.comboBox_6.currentText(),
            'path_to_init_checkpoint': self.ui.lineEdit_9.text(),
            'path_to_data': self.ui.lineEdit_10.text(),
            'epochs': self.ui.lineEdit_11.text(),
            'final_model_type': self.ui.comboBox_6.currentText(),
            'final_checkpoint_name': self.ui.lineEdit_16.text()
        }
        with open('train_config.json', 'w', encoding='utf8') as command_file:
            json.dump(config_dict, command_file)
        os.startfile('train_detector.py')

    def on_test_checkpoint_click(self):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Открыть файл', '')
        if filename:
            self.ui.lineEdit_12.setText(filename)

    def on_test_button_click(self):
        config_dict = {
            'model_type': self.ui.comboBox_14.currentText(),
            'path_to_init_checkpoint': self.ui.lineEdit_12.text(),
            'camera_index': self.ui.comboBox_13.currentIndex(),
            'conf': self.ui.lineEdit_15.text(),
            'iou': self.ui.lineEdit_14.text()
        }
        with open('test_config.json', 'w', encoding='utf8') as command_file:
            json.dump(config_dict, command_file)
        os.startfile('detector_test.py')


app = QtWidgets.QApplication([])
application = mywindow()
application.show()
sys.exit(app.exec())

