from PyQt5 import QtWidgets
from interface import Ui_MainWindow
import sys
import os
import json

from GestureDetector import GestureDetector
from CommandExecutor import CommandExecutor


class mywindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(mywindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.command_executor = CommandExecutor()
        self.ui.pushButton_2.clicked.connect(self.on_create_json_click)
        self.ui.pushButton.clicked.connect(self.on_open_json_click)
        self.ui.pushButton_4.clicked.connect(self.on_open_script_click)
        self.ui.pushButton_5.clicked.connect(self.on_save_command_click)
        self.ui.pushButton_6.clicked.connect(self.on_open_checkpoint_click)
        self.ui.pushButton_7.clicked.connect(self.on_start_click)


    def on_create_json_click(self):
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Создать файл', '', 'JSON Files (*.json)')
        if filename:
            self.ui.lineEdit.setText(filename)
            self.command_executor = CommandExecutor(filename)

    def on_open_json_click(self):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Открыть файл', '', 'JSON Files (*.json)')
        if filename:
            self.ui.lineEdit.setText(filename)
            self.command_executor = CommandExecutor(filename)

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
            self.command_executor.create_command([float(gesture),], script_path)
            msg.setIcon(QtWidgets.QMessageBox.Information)
            msg.setText("Команда успешно добавлена!")
        else:
            msg.setIcon(QtWidgets.QMessageBox.Critical)
            msg.setText("Python-скрипта по указанному пути не существует. Проверьте правильность введённого пути.")
        msg.setStandardButtons(QtWidgets.QMessageBox.Ok | QtWidgets.QMessageBox.Cancel)
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
            'conf': self.ui.lineEdit_4.text(),
            'iou': self.ui.lineEdit_5.text(),
        }
        with open('config.json', 'w', encoding='utf8') as command_file:
            json.dump(config_dict, command_file)
        os.startfile('test.py')


app = QtWidgets.QApplication([])
application = mywindow()
application.show()

sys.exit(app.exec())

