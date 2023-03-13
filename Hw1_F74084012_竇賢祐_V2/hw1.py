from PyQt5 import QtWidgets, QtCore
from Q1 import *
from Q2 import *
from Q3 import *
from Q4 import *

from controller import MainWindow_controller


if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow_controller()
    window.show()

    window.ui.pushButton_11.clicked.connect(Q11)
    window.ui.pushButton_12.clicked.connect(Q12)
    window.ui.pushButton_13.clicked.connect(Q13)
    window.ui.pushButton_14.clicked.connect(Q14)
    window.ui.pushButton_21.clicked.connect(Q21)
    window.ui.pushButton_22.clicked.connect(Q22)
    window.ui.pushButton_23.clicked.connect(Q23)
    window.ui.pushButton_31.clicked.connect(Q31)
    window.ui.pushButton_32.clicked.connect(Q32)
    window.ui.pushButton_33.clicked.connect(Q33)
    window.ui.pushButton_34.clicked.connect(Q34)
    window.ui.pushButton_41.clicked.connect(Q41)
    window.ui.pushButton_42.clicked.connect(Q42)
    window.ui.pushButton_43.clicked.connect(Q43)
    window.ui.pushButton_44.clicked.connect(Q44)
    sys.exit(app.exec_())