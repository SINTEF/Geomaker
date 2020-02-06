from PyQt5.QtCore import Qt, QAbstractListModel, QModelIndex, QVariant

from ..db import PROJECTS, db


class ProjectsModel(QAbstractListModel):
    """A list model for showing a list of projects. Suitable for use with
    QListView.setModel().
    """

    def rowCount(self, parent):
        return len(PROJECTS)

    def data(self, index, role):
        if role == Qt.DisplayRole:
            return QVariant(PROJECTS[index.row()][1])
        return QVariant()


class DatabaseModel(QAbstractListModel):
    """A list model for showing a list of regions. Suitable for use
    with QListView.setModel().
    """

    def __init__(self, interface):
        super().__init__()
        self.interface = interface
        db.notify(self)         # Ensure that we will be notified of changes

    def before_insert(self, index):
        self.beginInsertRows(QModelIndex(), index, index)

    def after_insert(self):
        self.endInsertRows()

    def before_delete(self, index):
        self.beginRemoveRows(QModelIndex(), index, index)

    def after_delete(self):
        self.endRemoveRows()

    def before_reset(self, lfid):
        self.interface.select_poly(-1)
        self._selected = lfid
        self.beginResetModel()

    def after_reset(self):
        self.endResetModel()
        self.interface.select_poly(self._selected)

    def data(self, index, role):
        if role == Qt.DisplayRole:
            return QVariant(db[index.row()].name)
        return QVariant()

    def setData(self, index, data, role):
        db.update_name(index.row(), data)
        return True

    def rowCount(self, parent):
        return len(db)

    def flags(self, index):
        return super().flags(index) | Qt.ItemIsEditable
