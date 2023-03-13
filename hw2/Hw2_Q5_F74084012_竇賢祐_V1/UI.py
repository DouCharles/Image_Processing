# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'untitled.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!
from Q5 import *
from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(254, 275)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(10, 10, 231, 221))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.pushButton5_1 = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.pushButton5_1.setObjectName("pushButton5_1")
        self.verticalLayout.addWidget(self.pushButton5_1)
        self.pushButton5_2 = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.pushButton5_2.setObjectName("pushButton5_2")
        self.verticalLayout.addWidget(self.pushButton5_2)
        self.pushButton5_3 = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.pushButton5_3.setObjectName("pushButton5_3")
        self.verticalLayout.addWidget(self.pushButton5_3)
        self.lineEdit = QtWidgets.QLineEdit(self.verticalLayoutWidget)
        self.lineEdit.setObjectName("lineEdit")
        self.verticalLayout.addWidget(self.lineEdit)
        self.pushButton5_4 = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.pushButton5_4.setObjectName("pushButton5_4")
        self.verticalLayout.addWidget(self.pushButton5_4)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 254, 21))
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
        self.pushButton5_1.setText(_translate("MainWindow", "1. Show Model Structure"))
        self.pushButton5_2.setText(_translate("MainWindow", "2. Show TensorBoard"))
        self.pushButton5_3.setText(_translate("MainWindow", "3. Test"))
        self.pushButton5_4.setText(_translate("MainWindow", "4. Data Augmantation"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)

    ui.pushButton5_1.clicked.connect(Q51)
    ui.pushButton5_2.clicked.connect(Q52)
    ui.pushButton5_3.clicked.connect(lambda:Q53(ui.lineEdit.text()))
    ui.pushButton5_4.clicked.connect(Q54)

    MainWindow.show()
    sys.exit(app.exec_())

