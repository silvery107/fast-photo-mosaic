# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Mode_C_Result.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_ModeCResultWindow(object):
    def setupUi(self, ModeCResultWindow):
        ModeCResultWindow.setObjectName("ModeCResultWindow")
        ModeCResultWindow.resize(800, 600)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/icon/pic/logo.ico"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        ModeCResultWindow.setWindowIcon(icon)
        self.label = QtWidgets.QLabel(ModeCResultWindow)
        self.label.setGeometry(QtCore.QRect(190, 20, 421, 51))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.Home_button = QtWidgets.QPushButton(ModeCResultWindow)
        self.Home_button.setGeometry(QtCore.QRect(220, 500, 121, 41))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        self.Home_button.setFont(font)
        self.Home_button.setObjectName("Home_button")
        self.exit_button = QtWidgets.QPushButton(ModeCResultWindow)
        self.exit_button.setGeometry(QtCore.QRect(440, 500, 121, 41))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        self.exit_button.setFont(font)
        self.exit_button.setObjectName("exit_button")
        self.label_2 = QtWidgets.QLabel(ModeCResultWindow)
        self.label_2.setGeometry(QtCore.QRect(90, 120, 601, 351))
        self.label_2.setText("")
        self.label_2.setPixmap(QtGui.QPixmap("logo.ico"))
        self.label_2.setScaledContents(True)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.classification_type = QtWidgets.QLabel(ModeCResultWindow)
        self.classification_type.setGeometry(QtCore.QRect(230, 70, 351, 31))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.classification_type.setFont(font)
        self.classification_type.setObjectName("classification_type")

        self.retranslateUi(ModeCResultWindow)
        self.Home_button.clicked.connect(ModeCResultWindow.next_button_click)
        self.exit_button.clicked.connect(ModeCResultWindow.exit_button_click)
        QtCore.QMetaObject.connectSlotsByName(ModeCResultWindow)

    def retranslateUi(self, ModeCResultWindow):
        _translate = QtCore.QCoreApplication.translate
        ModeCResultWindow.setWindowTitle(_translate("ModeCResultWindow", "Image Composite Result"))
        self.label.setText(_translate("ModeCResultWindow", "Class-specified Image Composite Result"))
        self.Home_button.setText(_translate("ModeCResultWindow", "Home"))
        self.exit_button.setText(_translate("ModeCResultWindow", "Exit"))
        self.classification_type.setText(_translate("ModeCResultWindow", "Classification Type:"))
import logo_icon_rc
