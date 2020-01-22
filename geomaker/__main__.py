import functools
from os.path import dirname, realpath, join
from operator import attrgetter
import sys

from PyQt5.QtCore import Qt, QObject, QUrl, pyqtSlot, QAbstractListModel, QModelIndex, QVariant
from PyQt5.QtWebChannel import QWebChannel
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtWidgets import QApplication, QWidget, QMainWindow, QVBoxLayout, QHBoxLayout, QListView, QLabel, QInputDialog

from geomaker.db import Database, Polygon


def label(text):
    label = QLabel()
    label.setText(text)
    return label


class DatabaseModel(QAbstractListModel):

    def __init__(self, db):
        super().__init__()
        self.db = db
        db.notify(self)

    def before_insert(self, index):
        self.beginInsertRows(QModelIndex(), index, index)

    def after_insert(self):
        self.endInsertRows()

    def before_delete(self, index):
        self.beginRemoveRows(QModelIndex(), index, index)

    def after_delete(self):
        self.endRemoveRows()

    def data(self, index, role):
        if role == Qt.DisplayRole:
            return QVariant(self.db[index.row()].name)
        return QVariant()

    def rowCount(self, parent):
        return len(self.db)


class JSInterface(QObject):

    def __init__(self, db, main):
        super().__init__()
        self.db = db
        self.main = main

    @pyqtSlot(int, str)
    def add_poly(self, lfid, data):
        name, accept = QInputDialog.getText(self.main, 'Name', 'Name this region:')
        if accept:
            self.db.create(lfid, name, data)

    @pyqtSlot(int, str)
    def edit_poly(self, lfid, data):
        self.db.update(lfid, data)

    @pyqtSlot(int)
    def remove_poly(self, lfid):
        self.db.delete(lfid)


class DatabaseWidget(QWidget):

    def __init__(self, db):
        super().__init__()
        self.db = db
        self.create_ui()

    def create_ui(self):
        box = QVBoxLayout()
        self.setLayout(box)

        box.addWidget(label('<strong>Stored regions</strong>'))

        listview = QListView()
        listview.setModel(DatabaseModel(self.db))
        box.addWidget(listview)


class MainWidget(QWidget):

    def __init__(self, db):
        super().__init__()
        self.db = db
        self.create_ui()

    def create_ui(self):
        box = QHBoxLayout()
        self.setLayout(box)

        # Web view
        self.view = QWebEngineView()
        self.view.loadFinished.connect(self.add_polys)

        self.interface = JSInterface(self.db, self)
        self.channel = QWebChannel()
        self.channel.registerObject("Main", self.interface)
        self.view.page().setWebChannel(self.channel)

        html = join(dirname(realpath(__file__)), "assets/map.html")
        self.view.setUrl(QUrl.fromLocalFile(html))
        box.addWidget(self.view)

        box.addWidget(DatabaseWidget(self.db))

    def add_polys(self):
        for poly in self.db:
            points = str(poly.points)
            self.view.page().runJavaScript(f'add_object({points})', poly.set_lfid)


class MainWindow(QMainWindow):

    def __init__(self, db):
        super().__init__()
        self.setWindowTitle('GeoMaker')
        self.setCentralWidget(MainWidget(db))


def main():
    db = Database()

    app = QApplication(sys.argv)
    win = MainWindow(db)
    win.showMaximized()
    sys.exit(app.exec_())
