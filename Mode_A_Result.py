# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Mode_A_Result.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_ModeAResultWindow(object):
    def setupUi(self, ModeAResultWindow):
        ModeAResultWindow.setObjectName("ModeAResultWindow")
        ModeAResultWindow.resize(800, 600)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/icon/pic/logo.ico"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        ModeAResultWindow.setWindowIcon(icon)
        self.label = QtWidgets.QLabel(ModeAResultWindow)
        self.label.setGeometry(QtCore.QRect(240, 30, 301, 51))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.Home_button = QtWidgets.QPushButton(ModeAResultWindow)
        self.Home_button.setGeometry(QtCore.QRect(220, 500, 121, 41))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        self.Home_button.setFont(font)
        self.Home_button.setObjectName("Home_button")
        self.exit_button = QtWidgets.QPushButton(ModeAResultWindow)
        self.exit_button.setGeometry(QtCore.QRect(440, 500, 121, 41))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        self.exit_button.setFont(font)
        self.exit_button.setObjectName("exit_button")
        self.label_2 = QtWidgets.QLabel(ModeAResultWindow)
        self.label_2.setGeometry(QtCore.QRect(90, 120, 601, 351))
        self.label_2.setText("")
        self.label_2.setPixmap(QtGui.QPixmap("logo.ico"))
        self.label_2.setScaledContents(True)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")

        self.retranslateUi(ModeAResultWindow)
        self.Home_button.clicked.connect(ModeAResultWindow.next_button_click)
        self.exit_button.clicked.connect(ModeAResultWindow.exit_button_click)
        QtCore.QMetaObject.connectSlotsByName(ModeAResultWindow)

    def retranslateUi(self, ModeAResultWindow):
        _translate = QtCore.QCoreApplication.translate
        ModeAResultWindow.setWindowTitle(_translate("ModeAResultWindow", "Image Composite Result"))
        self.label.setText(_translate("ModeAResultWindow", "Image Composite Result"))
        self.Home_button.setText(_translate("ModeAResultWindow", "Home"))
        self.exit_button.setText(_translate("ModeAResultWindow", "Exit"))
import logo_icon_rc
