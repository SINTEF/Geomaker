import functools
from os.path import dirname, realpath, join
import sys

from PyQt5.QtCore import QObject, QUrl, pyqtSlot
from PyQt5.QtWebChannel import QWebChannel
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtWidgets import QApplication, QWidget, QMainWindow, QVBoxLayout


class JSInterface(QObject):

    def __init__(self, main):
        super().__init__()
        self.main = main

    @pyqtSlot(int, str)
    def add_object(self, id, contents):
        print('add', id, contents)

    @pyqtSlot(int, str)
    def edit_object(self, id, contents):
        print('edit', id, contents)

    @pyqtSlot(int)
    def remove_object(self, id):
        print('remove', id)


class MainWidget(QWidget):

    def __init__(self):
        super().__init__()
        self.create_ui()

    def create_ui(self):
        vbox = QVBoxLayout()
        self.setLayout(vbox)

        view = QWebEngineView()

        self.interface = JSInterface(self)
        self.channel = QWebChannel()
        self.channel.registerObject("Main", self.interface)
        view.page().setWebChannel(self.channel)

        html = join(dirname(realpath(__file__)), "assets/map.html")
        view.setUrl(QUrl.fromLocalFile(html))
        vbox.addWidget(view)


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle('GeoMaker')
        self.setCentralWidget(MainWidget())


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.showMaximized()
    sys.exit(app.exec_())
