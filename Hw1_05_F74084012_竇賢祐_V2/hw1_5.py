from hw1_5UI import Ui_MainWindow
from PyQt5 import QtWidgets, QtCore
#import torch
from Q5 import *

class MainWindow_controller(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__() # in python3, super(Class, self).xxx = super().xxx
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)


if __name__ == '__main__':
    import sys

    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow_controller()
    window.show()

    window.ui.pushButton_1.clicked.connect(Q51)
    window.ui.pushButton_2.clicked.connect(Q52)
    window.ui.pushButton_3.clicked.connect(Q53)
    window.ui.pushButton_4.clicked.connect(Q54)
    window.ui.pushButton_5.clicked.connect(lambda : Q55(window.ui.lineEdit.text()))






    sys.exit(app.exec_())
