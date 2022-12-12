import sys
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit
from PyQt5.QtGui import QIntValidator
from main.py import App

k1 = 1.0
k2 = 1.0
image_path = 'images/'
image_name = 'image.png'
image2_name = 'image2.png'
output_name = 'output.png'

class AddWindow(QWidget):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Add Window")
        self.label = QLabel("Another Window")
        self.k1 = QLineEdit()
        self.k1.setText(str(k1))
        self.k1.move(20, 20)
        self.k1.resize(140,20)

        self.k2 = QLineEdit()
        self.k2.setText(str(k2))
        self.k2.move(20, 20)
        self.k2.resize(140,20)
        self.img1 = QLabel(image_name + " +")
        self.img2 = QLabel(image2_name)
        self.out = QLabel("= " + output_name)

        onlyInt = QIntValidator()
        onlyInt.setRange(-100, 100)
        self.k1.setValidator(onlyInt)
        self.k2.setValidator(onlyInt)


        self.k1.textChanged.connect(self.add)
        self.k2.textChanged.connect(self.add)

        hbox = QHBoxLayout()
        hbox.addWidget(self.k1)
        hbox.addWidget(self.img1)
        hbox.addWidget(self.k2)
        hbox.addWidget(self.img2)
        hbox.addWidget(self.out)

        self.setLayout(hbox)


    def add(self):
        App.add(k1, k2)
