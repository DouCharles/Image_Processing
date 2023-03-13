# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'hw2.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
from Q2 import *

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(239, 286)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(10, 10, 221, 231))
        self.groupBox.setObjectName("groupBox")
        self.verticalLayoutWidget = QtWidgets.QWidget(self.groupBox)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(0, 10, 221, 211))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.pushButton_2_1 = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.pushButton_2_1.setObjectName("pushButton_2_1")
        self.verticalLayout.addWidget(self.pushButton_2_1)
        self.pushButton_2_2 = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.pushButton_2_2.setObjectName("pushButton_2_2")
        self.verticalLayout.addWidget(self.pushButton_2_2)
        self.groupBox_2 = QtWidgets.QGroupBox(self.verticalLayoutWidget)
        self.groupBox_2.setObjectName("groupBox_2")
        self.label = QtWidgets.QLabel(self.groupBox_2)
        self.label.setGeometry(QtCore.QRect(20, 20, 71, 20))
        self.label.setObjectName("label")
        self.lineEdit = QtWidgets.QLineEdit(self.groupBox_2)
        self.lineEdit.setGeometry(QtCore.QRect(90, 20, 113, 20))
        self.lineEdit.setObjectName("lineEdit")
        self.pushButton_2_3 = QtWidgets.QPushButton(self.groupBox_2)
        self.pushButton_2_3.setGeometry(QtCore.QRect(30, 60, 161, 23))
        self.pushButton_2_3.setObjectName("pushButton_2_3")
        self.verticalLayout.addWidget(self.groupBox_2)
        self.pushButton_2_4 = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.pushButton_2_4.setObjectName("pushButton_2_4")
        self.verticalLayout.addWidget(self.pushButton_2_4)
        self.pushButton_2_5 = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.pushButton_2_5.setObjectName("pushButton_2_5")
        self.verticalLayout.addWidget(self.pushButton_2_5)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 239, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.groupBox.setTitle(_translate("MainWindow", "2.Calibration"))
        self.pushButton_2_1.setText(_translate("MainWindow", "2.1 Find Corners"))
        self.pushButton_2_2.setText(_translate("MainWindow", "2.2 Find Intrinsic"))
        self.groupBox_2.setTitle(_translate("MainWindow", "2.3Find Extrinsic"))
        self.label.setText(_translate("MainWindow", "Select image:"))
        self.pushButton_2_3.setText(_translate("MainWindow", "2.3 Find Extrinsic"))
        self.pushButton_2_4.setText(_translate("MainWindow", "2.4 Find Distortion"))
        self.pushButton_2_5.setText(_translate("MainWindow", "2.5 Show result"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    ui.pushButton_2_1.clicked.connect(Q21)
    ui.pushButton_2_2.clicked.connect(Q22)
    ui.pushButton_2_3.clicked.connect(lambda : Q23(ui.lineEdit.text()))
    ui.pushButton_2_4.clicked.connect(Q24)
    ui.pushButton_2_5.clicked.connect(Q25)

    MainWindow.show()
    sys.exit(app.exec_())

