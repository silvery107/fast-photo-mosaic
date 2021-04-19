# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Mode_B_Result.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_ModeBResultWindow(object):
    def setupUi(self, ModeBResultWindow):
        ModeBResultWindow.setObjectName("ModeBResultWindow")
        ModeBResultWindow.resize(400, 300)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/icon/pic/logo.ico"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        ModeBResultWindow.setWindowIcon(icon)
        self.label = QtWidgets.QLabel(ModeBResultWindow)
        self.label.setGeometry(QtCore.QRect(100, 10, 201, 51))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.Home_button = QtWidgets.QPushButton(ModeBResultWindow)
        self.Home_button.setGeometry(QtCore.QRect(60, 220, 121, 41))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        self.Home_button.setFont(font)
        self.Home_button.setObjectName("Home_button")
        self.exit_button = QtWidgets.QPushButton(ModeBResultWindow)
        self.exit_button.setGeometry(QtCore.QRect(210, 220, 121, 41))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        self.exit_button.setFont(font)
        self.exit_button.setObjectName("exit_button")
        self.label_result = QtWidgets.QLabel(ModeBResultWindow)
        self.label_result.setGeometry(QtCore.QRect(10, 70, 371, 131))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        self.label_result.setFont(font)
        self.label_result.setAlignment(QtCore.Qt.AlignCenter)
        self.label_result.setObjectName("label_result")

        self.retranslateUi(ModeBResultWindow)
        self.Home_button.clicked.connect(ModeBResultWindow.next_button_click)
        self.exit_button.clicked.connect(ModeBResultWindow.exit_button_click)
        QtCore.QMetaObject.connectSlotsByName(ModeBResultWindow)

    def retranslateUi(self, ModeBResultWindow):
        _translate = QtCore.QCoreApplication.translate
        ModeBResultWindow.setWindowTitle(_translate("ModeBResultWindow", "Classification Result"))
        self.label.setText(_translate("ModeBResultWindow", "Classification Result"))
        self.Home_button.setText(_translate("ModeBResultWindow", "Home"))
        self.exit_button.setText(_translate("ModeBResultWindow", "Exit"))
        self.label_result.setText(_translate("ModeBResultWindow", "<html><head/><body><p>Total Classified Images: </p><p>Correct: ;Error: </p><p>Correct rate: </p><p>Time cost:</p><p><br/></p></body></html>"))
import logo_icon_rc
