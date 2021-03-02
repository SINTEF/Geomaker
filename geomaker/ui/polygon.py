# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'gui/polygon.ui'
#
# Created by: PyQt5 UI code generator 5.14.1
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Polygon(object):
    def setupUi(self, Polygon):
        Polygon.setObjectName("Polygon")
        Polygon.resize(489, 403)
        Polygon.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.verticalLayout = QtWidgets.QVBoxLayout(Polygon)
        self.verticalLayout.setObjectName("verticalLayout")
        self.formLayout = QtWidgets.QFormLayout()
        self.formLayout.setObjectName("formLayout")
        self.nameLabel = QtWidgets.QLabel(Polygon)
        self.nameLabel.setObjectName("nameLabel")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.nameLabel)
        self.name = QtWidgets.QLineEdit(Polygon)
        self.name.setObjectName("name")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.name)
        self.coordinatesLabel = QtWidgets.QLabel(Polygon)
        self.coordinatesLabel.setObjectName("coordinatesLabel")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.coordinatesLabel)
        self.coordinates = QtWidgets.QComboBox(Polygon)
        self.coordinates.setObjectName("coordinates")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.coordinates)
        self.verticalLayout.addLayout(self.formLayout)
        self.points = QtWidgets.QPlainTextEdit(Polygon)
        font = QtGui.QFont()
        font.setFamily("Monospace")
        self.points.setFont(font)
        self.points.setObjectName("points")
        self.verticalLayout.addWidget(self.points)
        self.buttonBox = QtWidgets.QDialogButtonBox(Polygon)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.verticalLayout.addWidget(self.buttonBox)

        self.retranslateUi(Polygon)
        self.buttonBox.accepted.connect(Polygon.accept)
        self.buttonBox.rejected.connect(Polygon.reject)
        QtCore.QMetaObject.connectSlotsByName(Polygon)

    def retranslateUi(self, Polygon):
        _translate = QtCore.QCoreApplication.translate
        Polygon.setWindowTitle(_translate("Polygon", "New polygon"))
        self.nameLabel.setText(_translate("Polygon", "Name"))
        self.coordinatesLabel.setText(_translate("Polygon", "Coordinates"))
        self.points.setPlainText(_translate("Polygon", "Enter points in order, one per line\n"
"This is an example\n"
"0.13, -3.2"))
