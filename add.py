import sys
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit


class AddWindow(QWidget):

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self) :
        self.setWindowTitle("Add")
        self.textbox = QLineEdit(self)
        self.textbox.move(20, 20)
        self.textbox.resize(140,20)

        # self.lineEdit.textChanged.connect(self.add)

        hbox = QHBoxLayout()

    # def add(self):
    #     return cv
