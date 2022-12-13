import sys
import cv2
import os.path
import errno
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit, QComboBox
from PyQt5.QtGui import QPixmap, QImage, QColor, QDoubleValidator, QIntValidator
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt,QObject, QRect

# mpl.rcParams["text.usetex"] = True
# mpl.rcParams['text.latex.preamble'] = r'\usepackage{{amsmath}}'

image_path = 'images/'
image_name = 'image.png'
image2_name = 'image2.png'
output_name = 'output.png'

popup_size = QRect(100, 100, 400, 200)
output_size = 1024, 1024, 3

class App(QWidget):
    cv_img = None
    cv_img2 = None
    cv_out = np.zeros(output_size, dtype=np.uint8)
    app_width = 0
    app_height = 0

    cv_img_temp = None
    cv_img2_temp = None
    cv_out_temp = None

    def __init__(self, width, height):
        super().__init__()
        cv_imgs = read_img_from_file()
        self.cv_img = cv_imgs[0]
        self.cv_img2 = cv_imgs[1]
        self.app_width = width
        self.app_height = height

        self.initUI()

    # init UI elements
    def initUI(self):
        self.setWindowTitle("Matrix Math Visualizer")
        self.display_width = 640
        self.display_height = 480

        self.image = QLabel(self)
        self.image.setAlignment(Qt.AlignCenter)
        self.image_text = QLabel(image_name + " (" + str(self.cv_img.shape[0]) + "x" + str(self.cv_img.shape[1]) + ")")
        self.image_text.setAlignment(Qt.AlignCenter)

        self.image2 = QLabel(self)
        self.image2.setAlignment(Qt.AlignCenter)
        self.image2_text = QLabel(image2_name + " (" + str(self.cv_img2.shape[0]) + "x" + str(self.cv_img2.shape[1]) + ")")
        self.image2_text.setAlignment(Qt.AlignCenter)

        self.output = QLabel(self)
        self.output.setAlignment(Qt.AlignCenter)
        self.output_text = QLabel(output_name + " (" + str(self.cv_out.shape[0]) + "x" + str(self.cv_out.shape[1]) + ")")
        self.output_text.setAlignment(Qt.AlignCenter)

        # self.setFixedWidth(self.display_width)

        self.color_button = QPushButton("Grayscale")
        self.color_button.clicked.connect(self.color_button_f)

        blur_button = QPushButton("Kernel Blur...")
        blur_button.clicked.connect(self.blur_button)

        add_button = QPushButton("Linear Combination...")
        add_button.clicked.connect(self.add_button)

        mult_button = QPushButton("Element-Wise Multiplication")
        mult_button.clicked.connect(self.mult_button)

        matrix_mult_button = QPushButton("Dot Product")
        matrix_mult_button.clicked.connect(self.mult_button)

        bitwise_button = QPushButton("Bitwise Operations...")
        bitwise_button.clicked.connect(self.bitwise_button)

        transpose_button = QPushButton("Transpose")
        transpose_button.clicked.connect(self.transpose_button)

        rotate_button = QPushButton("Rotate...")
        rotate_button.clicked.connect(self.rotate_button)

        reset_button = QPushButton("Reset")
        reset_button.clicked.connect(self.reset_button)

        # self.w = QLabel(self)
        # self.w.resize(self.display_width, self.display_height)
        # mathTex_label = r'$\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$'
        # mathTex_label = '$C_{soil}=(1 - n) C_m + \\theta_w C_w$'
        # self.w.setPixmap(self.mathTex_to_QPixmap(mathTex_label, 12))

        # create a vertical box layout and add the two labels
        image_hbox = QHBoxLayout()
        image_vlabel = QVBoxLayout()
        image2_vlabel = QVBoxLayout()
        output_vlabel = QVBoxLayout()
        vbox = QVBoxLayout()
        vbox.addLayout(image_hbox)
        image_hbox.addLayout(image_vlabel)
        image_vlabel.addWidget(self.image)
        image_vlabel.addWidget(self.image_text)
        image_hbox.addLayout(image2_vlabel)
        image2_vlabel.addWidget(self.image2)
        image2_vlabel.addWidget(self.image2_text)
        image_hbox.addLayout(output_vlabel)
        output_vlabel.addWidget(self.output)
        output_vlabel.addWidget(self.output_text)
        vbox.addWidget(self.color_button)
        vbox.addWidget(blur_button)
        vbox.addWidget(add_button)
        vbox.addWidget(mult_button)
        vbox.addWidget(matrix_mult_button)
        vbox.addWidget(bitwise_button)
        vbox.addWidget(rotate_button)
        vbox.addWidget(transpose_button)
        vbox.addWidget(reset_button)
        # set the vbox layout as the widgets layout
        self.setLayout(vbox)

        # display converted image
        self.update_img()
        self.update_img2()
        self.update_out()

    # button callback
    def color_button_f(self):
        print("Swapping colorspaces")

        if len(self.cv_img.shape) == 3:
            self.cv_img = cv2.cvtColor(self.cv_img, cv2.COLOR_BGR2GRAY)
            self.cv_img2 = cv2.cvtColor(self.cv_img2, cv2.COLOR_BGR2GRAY)
            self.cv_out = cv2.cvtColor(self.cv_out, cv2.COLOR_BGR2GRAY)
            self.color_button.setText('Color (Reset)')
        else :
            self.reset_button()
            self.color_button.setText('Grayscale')

        self.update_img()
        self.update_img2()
        self.update_out()

    # button callback
    def rotate_button(self):
        print("Rotating")

        self.cv_img_temp = self.cv_img
        self.cv_img2_temp = self.cv_img2
        self.cv_out_temp = self.cv_out

        self.w = RotateWindow()
        self.w.setGeometry(popup_size)
        self.w.show()

    # function callback
    def rotate(self, theta, cv_img):
        cols = self.cv_out.shape[0]
        rows = self.cv_out.shape[1]

        M = cv2.getRotationMatrix2D((cols/2,rows/2), int(float(theta)), 1)

        if cv_img == image_name:
            self.cv_img2 = self.cv_img2_temp
            self.cv_out = self.cv_out_temp
            self.update_img2()
            self.update_out()

            self.cv_img = cv2.warpAffine(self.cv_img_temp, M, (cols,rows))
            self.update_img()
        elif cv_img == image2_name:
            self.cv_img = self.cv_img_temp
            self.cv_out = self.cv_out_temp
            self.update_img()
            self.update_out()

            self.cv_img2 = cv2.warpAffine(self.cv_img2_temp, M, (cols,rows))
            self.update_img2()
        else :
            self.cv_img = self.cv_img_temp
            self.cv_img2 = self.cv_img2_temp
            self.update_img()
            self.update_img2()

            self.cv_out = cv2.warpAffine(self.cv_out_temp, M, (cols,rows))
            self.update_out()

    # button callback
    def blur_button(self):
        print("Blurring")

        self.w = BlurWindow()
        self.w.setGeometry(popup_size)
        self.w.show()

    # function callback
    def blur(self, kernel_size, img_select):
        if img_select == image_name:
            self.cv_img = cv2.blur(self.cv_img, kernel_size)
            self.update_img()
        elif img_select == image2_name:
            self.cv_img2 = cv2.blur(self.cv_img2, kernel_size)
            self.update_img2()
        else :
            self.cv_out = cv2.blur(self.cv_out, kernel_size)
            self.update_out()


    # button callback
    def add_button(self):
        print("Linear Combination")

        self.w = AddWindow()
        self.w.setGeometry(popup_size)
        self.w.show()

    # function callback
    def add(self, k1, k2):
        self.cv_out = k1 * self.cv_img + k2 * self.cv_img2
        self.cv_out = np.around(self.cv_out).astype(np.uint8)
        self.update_out()

    # button callback
    def mult_button(self):
        print("Element-Wise Multiplication")
        self.cv_out = np.multiply(self.cv_img, self.cv_img2)
        self.update_out()

    # button callback
    def mult_button(self):
        print("Dot Product")
        if len(self.cv_img.shape) == 3:
            bo,go,ro = cv2.split(self.cv_out)
            b1,g1,r1 = cv2.split(self.cv_img)
            b2,g2,r2 = cv2.split(self.cv_img2)
            bo = np.dot(b1, b2)
            go = np.dot(g1, g2)
            ro = np.dot(r1, r2)

            self.cv_out = cv2.merge([bo,go,ro])
        else:
            self.cv_out = np.dot(self.cv_img, self.cv_img2)

        self.update_out()

    # button callback
    def bitwise_button(self):
        print("Bitwise Operation")

        self.w = BitwiseWindow()
        self.w.setGeometry(popup_size)
        self.w.show()

    # function callback
    def bitwise(self, cv_str, cv_str2, operation):
        cv_img = None
        cv_img2 = None

        if cv_str == image_name:
            cv_img = self.cv_img
        elif cv_str == image2_name:
            cv_img = self.cv_img2
        else:
            cv_img = self.cv_out

        if cv_str2 == image_name:
            cv_img2 = self.cv_img
        elif cv_str2 == image2_name:
            cv_img2 = self.cv_img2
        else:
            cv_img2 = self.cv_out

        if operation == "AND":
            self.cv_out = cv2.bitwise_and(cv_img, cv_img2)
        elif operation == "OR":
            self.cv_out = cv2.bitwise_or(cv_img, cv_img2)
        else:
            self.cv_out = cv2.bitwise_xor(cv_img, cv_img2)

        self.update_out()

    # button callback
    def transpose_button(self):
        print("Image transposed")
        self.cv_out = cv2.transpose(self.cv_out)
        self.update_out()

    # button callback
    def reset_button(self):
        print("Image reset")
        self.cv_img = read_img_from_file()[0]
        self.cv_img2 = read_img_from_file()[1]
        self.cv_out = np.zeros(output_size, dtype=np.uint8)

        self.update_img()
        self.update_img2()
        self.update_out()

    # update q
    def update_img(self):
        self.image.setPixmap(self.convert_cv_qt(self.cv_img))
        self.image_text = QLabel(image_name + " (" + str(self.cv_img.shape[0]) + "x" + str(self.cv_img.shape[1]) + ")")
        self.image_text.setAlignment(Qt.AlignCenter)

    def update_img2(self):
        self.image2.setPixmap(self.convert_cv_qt(self.cv_img2))
        self.image2_text = QLabel(image2_name + " (" + str(self.cv_img2.shape[0]) + "x" + str(self.cv_img2.shape[1]) + ")")
        self.image2_text.setAlignment(Qt.AlignCenter)

    def update_out(self):
        self.output.setPixmap(self.convert_cv_qt(self.cv_out))
        self.output_text = QLabel(output_name + " (" + str(self.cv_out.shape[0]) + "x" + str(self.cv_out.shape[1]) + ")")
        self.output_text.setAlignment(Qt.AlignCenter)
        write_img_to_file(self.cv_out)

    # convert openCV mat to to Qt QPixmap
    def convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(int(self.app_width / 4), int(self.app_width / 4), Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)


def read_img_from_file() :
    print("Loading images...")

    image_loc = image_path + image_name
    image2_loc = image_path + image2_name
    # load image
    if not os.path.isfile(image_loc) :
        print("Missing image file of name", image_name)
        raise OSError(errno.ENOENT, "Missing image file at", image_loc)

    if not os.path.isfile(image2_loc) :
        print("Missing image file of name", image2_name)
        raise OSError(errno.ENOENT, "Missing image file at", image2_loc)

    return cv2.imread(image_loc, cv2.IMREAD_ANYCOLOR), cv2.imread(image2_loc, cv2.IMREAD_ANYCOLOR)

def write_img_to_file(cv_out) :
    cv2.imwrite(image_path + output_name, cv_out)



class AddWindow(QWidget):
    k1 = 1
    k2 = 1

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Linear Combination")
        self.k1Line = QLineEdit()
        self.k1Line.setText(str(self.k1))
        self.k1Line.move(20, 20)
        self.k1Line.resize(140,20)

        self.k2Line = QLineEdit()
        self.k2Line.setText(str(self.k2))
        self.k2Line.move(20, 20)
        self.k2Line.resize(140,20)
        self.img1 = QLabel(image_name + " +")
        self.img2 = QLabel(image2_name)
        self.out = QLabel("= " + output_name)

        self.k1Line.textChanged.connect(self.add)
        self.k2Line.textChanged.connect(self.add)

        hbox = QHBoxLayout()
        hbox.addWidget(self.k1Line)
        hbox.addWidget(self.img1)
        hbox.addWidget(self.k2Line)
        hbox.addWidget(self.img2)
        hbox.addWidget(self.out)

        self.setLayout(hbox)

        self.add()


    def add(self):
        if (self.k1Line.text() == ''):
            self.k1 = 0
        else :
            try:
                self.k1 = float(self.k1Line.text())
            except ValueError:
                print('Please enter a float in field 1')

        if (self.k2Line.text() == ''):
            self.k2 = 0
        else :
            try:
                self.k2 = float(self.k2Line.text())
            except ValueError:
                print('Please enter a float in field 2')

        a.add(self.k1, self.k2)

class BlurWindow(QWidget):
    k1 = 5
    k2 = 5

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Blur")
        self.k1Line = QLineEdit()
        self.k1Line.setText(str(self.k1))
        self.k1Line.move(20, 20)
        self.k1Line.resize(140,20)

        self.k2Line = QLineEdit()
        self.k2Line.setText(str(self.k2))
        self.k2Line.move(20, 20)
        self.k2Line.resize(140,20)

        onlyInt = QIntValidator()
        onlyInt.setRange(0, 10000)
        self.k1Line.setValidator(onlyInt)
        self.k2Line.setValidator(onlyInt)

        self.img_select = QComboBox()
        self.img_select.addItem(image_name)
        self.img_select.addItem(image2_name)
        self.img_select.addItem(output_name)

        render_button = QPushButton("Render")
        render_button.clicked.connect(self.blur)

        hbox = QHBoxLayout()
        hbox.addWidget(self.k1Line)
        hbox.addWidget(self.k2Line)
        hbox.addWidget(self.img_select)
        hbox.addWidget(render_button)

        self.setLayout(hbox)

        self.blur()


    def blur(self):
        if (self.k1Line.text() == '' or self.k1Line.text() == '0'):
            self.k1 = 1
        else :
            try:
                self.k1 = int(self.k1Line.text())
            except ValueError:
                print('Please enter an int')

        if (self.k2Line.text() == '' or self.k2Line.text() == '0'):
            self.k2 = 1
        else :
            try:
                self.k2 = int(self.k2Line.text())
            except ValueError:
                print('Please enter an int')

        a.blur((self.k1, self.k2), self.img_select.currentText())

class BitwiseWindow(QWidget):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bitwise Operation")

        self.img_select = QComboBox()
        self.img_select.addItem(image_name)
        self.img_select.addItem(image2_name)
        self.img_select.addItem(output_name)

        self.img2_select = QComboBox()
        self.img2_select.addItem(image_name)
        self.img2_select.addItem(image2_name)
        self.img2_select.addItem(output_name)
        self.img2_select.setCurrentIndex(1)

        self.operator_select = QComboBox()
        self.operator_select.addItem("AND")
        self.operator_select.addItem("OR")
        self.operator_select.addItem("XOR")

        self.out = QLabel("= " + output_name)

        self.img_select.currentTextChanged.connect(self.bitwise)
        self.img2_select.currentTextChanged.connect(self.bitwise)
        self.operator_select.currentTextChanged.connect(self.bitwise)


        hbox = QHBoxLayout()
        hbox.addWidget(self.img_select)
        hbox.addWidget(self.operator_select)
        hbox.addWidget(self.img2_select)
        hbox.addWidget(self.out)

        self.setLayout(hbox)

        self.bitwise()


    def bitwise(self):
        a.bitwise(self.img_select.currentText(), self.img2_select.currentText(), self.operator_select.currentText())

class RotateWindow(QWidget):
    theta = 30.0

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Rotation")

        self.img_select = QComboBox()
        self.img_select.addItem(image_name)
        self.img_select.addItem(image2_name)
        self.img_select.addItem(output_name)

        self.thetaLine = QLineEdit()
        self.thetaLine.setText(str(self.theta))
        self.thetaLine.move(20, 20)
        self.thetaLine.resize(140,20)

        self.img_select.currentTextChanged.connect(self.rotate)
        self.thetaLine.textChanged.connect(self.rotate)


        hbox = QHBoxLayout()
        hbox.addWidget(self.thetaLine)
        hbox.addWidget(self.img_select)

        self.setLayout(hbox)

        self.rotate()


    def rotate(self):
        if (self.thetaLine.text() == ''):
            self.theta = 0
        else :
            try:
                self.theta = float(self.thetaLine.text())
            except ValueError:
                print('Please enter a float in field 1')

        a.rotate(self.theta, self.img_select.currentText())

def main() :
    app = QApplication(sys.argv)

    screen = app.primaryScreen()
    rect = screen.availableGeometry()
    global a
    a = App(rect.width(), rect.height())
    print("--> Running Matrix Math Visualizer")
    print("--> Source Code: https://github.com/dgorbunov")

    a.show()
    sys.exit(app.exec_())


if __name__=="__main__":
    main()
