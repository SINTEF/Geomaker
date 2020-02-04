# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'gui/thumbnail.ui'
#
# Created by: PyQt5 UI code generator 5.14.1
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Thumbnail(object):
    def setupUi(self, Thumbnail):
        Thumbnail.setObjectName("Thumbnail")
        Thumbnail.resize(400, 263)
        self.gridLayout = QtWidgets.QGridLayout(Thumbnail)
        self.gridLayout.setObjectName("gridLayout")
        self.thumbnail = QResizeLabel(Thumbnail)
        self.thumbnail.setText("")
        self.thumbnail.setAlignment(QtCore.Qt.AlignCenter)
        self.thumbnail.setObjectName("thumbnail")
        self.gridLayout.addWidget(self.thumbnail, 0, 0, 1, 1)

        self.retranslateUi(Thumbnail)
        QtCore.QMetaObject.connectSlotsByName(Thumbnail)

    def retranslateUi(self, Thumbnail):
        _translate = QtCore.QCoreApplication.translate
        Thumbnail.setWindowTitle(_translate("Thumbnail", "Form"))
from geomaker.ui.widgets import QResizeLabel
