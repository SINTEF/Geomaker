# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'gui/jobdialog.ui'
#
# Created by: PyQt5 UI code generator 5.14.1
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_JobDialog(object):
    def setupUi(self, JobDialog):
        JobDialog.setObjectName("JobDialog")
        JobDialog.resize(618, 224)
        self.gridLayout = QtWidgets.QGridLayout(JobDialog)
        self.gridLayout.setObjectName("gridLayout")
        self.groupBox = QtWidgets.QGroupBox(JobDialog)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(3)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupBox.sizePolicy().hasHeightForWidth())
        self.groupBox.setSizePolicy(sizePolicy)
        self.groupBox.setObjectName("groupBox")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.groupBox)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.email = QtWidgets.QLineEdit(self.groupBox)
        self.email.setObjectName("email")
        self.gridLayout_2.addWidget(self.email, 0, 1, 1, 1)
        self.label = QtWidgets.QLabel(self.groupBox)
        self.label.setObjectName("label")
        self.gridLayout_2.addWidget(self.label, 0, 0, 1, 1)
        self.widget = QtWidgets.QWidget(self.groupBox)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.widget.sizePolicy().hasHeightForWidth())
        self.widget.setSizePolicy(sizePolicy)
        self.widget.setObjectName("widget")
        self.gridLayout_2.addWidget(self.widget, 2, 0, 1, 2)
        self.dedicated = QtWidgets.QCheckBox(self.groupBox)
        self.dedicated.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.dedicated.setObjectName("dedicated")
        self.gridLayout_2.addWidget(self.dedicated, 1, 0, 1, 2)
        self.gridLayout.addWidget(self.groupBox, 0, 1, 1, 1)
        self.buttonBox = QtWidgets.QDialogButtonBox(JobDialog)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.gridLayout.addWidget(self.buttonBox, 1, 0, 1, 2)
        self.projectlist = QSelectedListView(JobDialog)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.projectlist.sizePolicy().hasHeightForWidth())
        self.projectlist.setSizePolicy(sizePolicy)
        self.projectlist.setObjectName("projectlist")
        self.gridLayout.addWidget(self.projectlist, 0, 0, 1, 1)

        self.retranslateUi(JobDialog)
        self.buttonBox.accepted.connect(JobDialog.accept)
        self.buttonBox.rejected.connect(JobDialog.reject)
        QtCore.QMetaObject.connectSlotsByName(JobDialog)

    def retranslateUi(self, JobDialog):
        _translate = QtCore.QCoreApplication.translate
        JobDialog.setWindowTitle(_translate("JobDialog", "Start new job"))
        self.groupBox.setTitle(_translate("JobDialog", "Job options"))
        self.label.setText(_translate("JobDialog", "E-mail address"))
        self.dedicated.setText(_translate("JobDialog", "Download dedicated file"))
from geomaker.ui.widgets import QSelectedListView
