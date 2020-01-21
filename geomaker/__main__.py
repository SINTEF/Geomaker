import functools
from os.path import dirname, realpath, join
import sys

from PyQt5.QtCore import QObject, QUrl, pyqtSlot
from PyQt5.QtWebChannel import QWebChannel
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtWidgets import QApplication, QWidget, QMainWindow, QVBoxLayout

from geomaker.db import Database


class JSInterface(QObject):

    def __init__(self, db):
        super().__init__()
        self.db = db

    @pyqtSlot(int, str)
    def add_poly(self, lfid, data):
        self.db.create(lfid, data)

    @pyqtSlot(int, str)
    def edit_poly(self, lfid, data):
        self.db.update(lfid, data)

    @pyqtSlot(int)
    def remove_poly(self, lfid):
        self.db.delete(lfid)


class MainWidget(QWidget):

    def __init__(self, db):
        super().__init__()
        self.db = db
        self.create_ui()

    def create_ui(self):
        vbox = QVBoxLayout()
        self.setLayout(vbox)

        self.view = QWebEngineView()
        self.view.loadFinished.connect(self.add_polys)

        self.interface = JSInterface(self.db)
        self.channel = QWebChannel()
        self.channel.registerObject("Main", self.interface)
        self.view.page().setWebChannel(self.channel)

        html = join(dirname(realpath(__file__)), "assets/map.html")
        self.view.setUrl(QUrl.fromLocalFile(html))
        vbox.addWidget(self.view)

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
