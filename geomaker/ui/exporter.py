# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'gui/exporter.ui'
#
# Created by: PyQt5 UI code generator 5.14.1
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Exporter(object):
    def setupUi(self, Exporter):
        Exporter.setObjectName("Exporter")
        Exporter.resize(829, 479)
        self.gridLayout_2 = QtWidgets.QGridLayout(Exporter)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.buttonBox = QtWidgets.QDialogButtonBox(Exporter)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.gridLayout.addWidget(self.buttonBox, 2, 0, 1, 1)
        self.widget = QtWidgets.QWidget(Exporter)
        self.widget.setObjectName("widget")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.widget)
        self.gridLayout_3.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.widget_2 = QtWidgets.QWidget(self.widget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.widget_2.sizePolicy().hasHeightForWidth())
        self.widget_2.setSizePolicy(sizePolicy)
        self.widget_2.setObjectName("widget_2")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.widget_2)
        self.gridLayout_4.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.label_2 = QtWidgets.QLabel(self.widget_2)
        self.label_2.setObjectName("label_2")
        self.gridLayout_4.addWidget(self.label_2, 4, 0, 1, 1)
        self.filename = QtWidgets.QComboBox(self.widget_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.filename.sizePolicy().hasHeightForWidth())
        self.filename.setSizePolicy(sizePolicy)
        self.filename.setEditable(True)
        self.filename.setObjectName("filename")
        self.gridLayout_4.addWidget(self.filename, 0, 1, 1, 1)
        self.colormaps = QtWidgets.QComboBox(self.widget_2)
        self.colormaps.setObjectName("colormaps")
        self.gridLayout_4.addWidget(self.colormaps, 6, 1, 1, 2)
        self.resolution = QtWidgets.QDoubleSpinBox(self.widget_2)
        self.resolution.setDecimals(1)
        self.resolution.setMaximum(1000.0)
        self.resolution.setSingleStep(0.1)
        self.resolution.setObjectName("resolution")
        self.gridLayout_4.addWidget(self.resolution, 4, 1, 1, 2)
        self.widget_3 = QtWidgets.QWidget(self.widget_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.widget_3.sizePolicy().hasHeightForWidth())
        self.widget_3.setSizePolicy(sizePolicy)
        self.widget_3.setObjectName("widget_3")
        self.gridLayout_4.addWidget(self.widget_3, 8, 0, 1, 3)
        self.Location = QtWidgets.QLabel(self.widget_2)
        self.Location.setObjectName("Location")
        self.gridLayout_4.addWidget(self.Location, 0, 0, 1, 1)
        self.label_4 = QtWidgets.QLabel(self.widget_2)
        self.label_4.setObjectName("label_4")
        self.gridLayout_4.addWidget(self.label_4, 6, 0, 1, 1)
        self.label = QtWidgets.QLabel(self.widget_2)
        self.label.setObjectName("label")
        self.gridLayout_4.addWidget(self.label, 1, 0, 1, 1)
        self.browsebtn = QtWidgets.QPushButton(self.widget_2)
        self.browsebtn.setObjectName("browsebtn")
        self.gridLayout_4.addWidget(self.browsebtn, 0, 2, 1, 1)
        self.formats = QtWidgets.QComboBox(self.widget_2)
        self.formats.setObjectName("formats")
        self.gridLayout_4.addWidget(self.formats, 1, 1, 1, 2)
        self.coordinates = QtWidgets.QComboBox(self.widget_2)
        self.coordinates.setObjectName("coordinates")
        self.gridLayout_4.addWidget(self.coordinates, 2, 1, 1, 2)
        self.label_3 = QtWidgets.QLabel(self.widget_2)
        self.label_3.setObjectName("label_3")
        self.gridLayout_4.addWidget(self.label_3, 2, 0, 1, 1)
        self.zero_sea_level = QtWidgets.QCheckBox(self.widget_2)
        self.zero_sea_level.setObjectName("zero_sea_level")
        self.gridLayout_4.addWidget(self.zero_sea_level, 7, 1, 1, 2)
        self.widget_4 = QtWidgets.QWidget(self.widget_2)
        self.widget_4.setMinimumSize(QtCore.QSize(0, 0))
        self.widget_4.setObjectName("widget_4")
        self.gridLayout_5 = QtWidgets.QGridLayout(self.widget_4)
        self.gridLayout_5.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_5.setHorizontalSpacing(0)
        self.gridLayout_5.setVerticalSpacing(6)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.widget_5 = QtWidgets.QWidget(self.widget_4)
        self.widget_5.setMinimumSize(QtCore.QSize(0, 50))
        self.widget_5.setObjectName("widget_5")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.widget_5)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setSpacing(0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.no_rot = QtWidgets.QRadioButton(self.widget_5)
        self.no_rot.setObjectName("no_rot")
        self.verticalLayout_2.addWidget(self.no_rot)
        self.north_rot = QtWidgets.QRadioButton(self.widget_5)
        self.north_rot.setObjectName("north_rot")
        self.verticalLayout_2.addWidget(self.north_rot)
        self.free_rot = QtWidgets.QRadioButton(self.widget_5)
        self.free_rot.setObjectName("free_rot")
        self.verticalLayout_2.addWidget(self.free_rot)
        self.gridLayout_5.addWidget(self.widget_5, 0, 1, 1, 1)
        self.widget_6 = QtWidgets.QWidget(self.widget_4)
        self.widget_6.setMinimumSize(QtCore.QSize(0, 20))
        self.widget_6.setObjectName("widget_6")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.widget_6)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setSpacing(0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.interior_bnd = QtWidgets.QRadioButton(self.widget_6)
        self.interior_bnd.setChecked(True)
        self.interior_bnd.setObjectName("interior_bnd")
        self.verticalLayout.addWidget(self.interior_bnd)
        self.exterior_bnd = QtWidgets.QRadioButton(self.widget_6)
        self.exterior_bnd.setObjectName("exterior_bnd")
        self.verticalLayout.addWidget(self.exterior_bnd)
        self.actual_bnd = QtWidgets.QRadioButton(self.widget_6)
        self.actual_bnd.setObjectName("actual_bnd")
        self.verticalLayout.addWidget(self.actual_bnd)
        self.gridLayout_5.addWidget(self.widget_6, 0, 0, 1, 1)
        self.fitwarning = QtWidgets.QLabel(self.widget_4)
        self.fitwarning.setStyleSheet("color: red;")
        self.fitwarning.setText("")
        self.fitwarning.setObjectName("fitwarning")
        self.gridLayout_5.addWidget(self.fitwarning, 3, 0, 1, 2)
        self.gridLayout_4.addWidget(self.widget_4, 5, 1, 1, 2)
        self.gridLayout_3.addWidget(self.widget_2, 0, 2, 1, 1)
        self.gridLayout.addWidget(self.widget, 1, 0, 1, 1)
        self.gridLayout_2.addLayout(self.gridLayout, 0, 0, 1, 1)

        self.retranslateUi(Exporter)
        self.buttonBox.accepted.connect(Exporter.accept)
        self.buttonBox.rejected.connect(Exporter.reject)
        QtCore.QMetaObject.connectSlotsByName(Exporter)

    def retranslateUi(self, Exporter):
        _translate = QtCore.QCoreApplication.translate
        Exporter.setWindowTitle(_translate("Exporter", "Export data"))
        self.label_2.setText(_translate("Exporter", "Resolution:"))
        self.resolution.setSuffix(_translate("Exporter", "m"))
        self.Location.setText(_translate("Exporter", "File:"))
        self.label_4.setText(_translate("Exporter", "Color map:"))
        self.label.setText(_translate("Exporter", "Format:"))
        self.browsebtn.setText(_translate("Exporter", "Browse..."))
        self.label_3.setText(_translate("Exporter", "Coordinates:"))
        self.zero_sea_level.setText(_translate("Exporter", "Zero at sea level"))
        self.no_rot.setText(_translate("Exporter", "Axis-aligned "))
        self.north_rot.setText(_translate("Exporter", "North-aligned"))
        self.free_rot.setText(_translate("Exporter", "Free rotation"))
        self.interior_bnd.setText(_translate("Exporter", "Largest interior rectangle"))
        self.exterior_bnd.setText(_translate("Exporter", "Smallest exterior rectangle (bounding box)"))
        self.actual_bnd.setText(_translate("Exporter", "True boundaries"))