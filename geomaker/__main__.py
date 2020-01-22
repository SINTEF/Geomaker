import functools
from os.path import dirname, realpath, join
from operator import attrgetter
import sys

from PyQt5.QtCore import (
    Qt, QObject, QUrl, pyqtSlot, QAbstractListModel, QModelIndex, QVariant, QItemSelectionModel
)
from PyQt5.QtWebChannel import QWebChannel
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtWidgets import (
    QApplication, QWidget, QMainWindow, QVBoxLayout, QGridLayout, QListView, QLabel, QInputDialog, QSplitter
)

from geomaker.db import Database, Polygon


def label(text):
    label = QLabel()
    label.setText(text)
    return label


def angle_to_degrees(angle, directions):
    direction = directions[1 if angle > 0 else 0]
    angle = abs(angle)
    degrees, angle = divmod(angle, 1.0)
    minutes, angle = divmod(60 * angle, 1.0)
    seconds = 60 * angle
    return f"{degrees:.0f}° {minutes:.0f}' {seconds:.3f}'' {direction}"


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

    def __init__(self, main):
        super().__init__()
        self.db = main.db
        self.db_widget = main.db_widget
        self.main = main

    @pyqtSlot(int, str)
    def add_poly(self, lfid, data):
        name, accept = QInputDialog.getText(self.main, 'Name', 'Name this region:')
        if accept:
            self.db.create(lfid, name, data)
            self.select_poly()
            self.select_poly(lfid)

    @pyqtSlot(int, str)
    def edit_poly(self, lfid, data):
        self.db.update(lfid, data)

    @pyqtSlot(int)
    def remove_poly(self, lfid):
        self.db.delete(lfid)

    @pyqtSlot(int)
    def select_poly(self, lfid=-1):
        if lfid < 0:
            self.db_widget.unselect()
        else:
            self.db_widget.select(self.db.index_of(lfid=lfid))


class PolyWidget(QWidget):

    def __init__(self):
        super().__init__()
        self.setVisible(False)
        self.create_ui()

    def _add_row(self, attrname, title, widget):
        self._rows += 1
        self.box.addWidget(label(title), self._rows, 0, Qt.AlignRight)
        self.box.addWidget(widget, self._rows, 1, Qt.AlignLeft)
        setattr(self, attrname, widget)

    def create_ui(self):
        self.box = QGridLayout()
        self.setLayout(self.box)
        self._rows = 0

        self.name = label('')
        self.box.addWidget(self.name, 0, 0, 1, 2, Qt.AlignCenter)

        self._add_row('west', 'West', label(''))
        self._add_row('east', 'East', label(''))
        self._add_row('south', 'South', label(''))
        self._add_row('north', 'North', label(''))

        self.box.setRowStretch(self._rows + 1, 1)

    def show(self, poly=None):
        if poly is None:
            self.setVisible(False)
            return
        self.setVisible(True)
        self.name.setText(f'<strong>{poly.name}</strong>')
        self.west.setText(f'{poly.west:.4f} ({angle_to_degrees(poly.west, "WE")})')
        self.east.setText(f'{poly.east:.4f} ({angle_to_degrees(poly.east, "WE")})')
        self.south.setText(f'{poly.south:.4f} ({angle_to_degrees(poly.south, "SN")})')
        self.north.setText(f'{poly.north:.4f} ({angle_to_degrees(poly.north, "SN")})')


class DatabaseWidget(QSplitter):

    def __init__(self, main):
        super().__init__()
        self.setOrientation(Qt.Vertical)
        self.main = main
        self.db = main.db
        self.create_ui()

    def create_ui(self):
        top = QWidget()
        box = QVBoxLayout()
        top.setLayout(box)

        box.addWidget(label('<strong>Stored regions</strong>'))

        self.listview = QListView()
        self.listview.setModel(DatabaseModel(self.db))
        self.listview.selectionModel().selectionChanged.connect(self.selection_changed)
        self.listview.doubleClicked.connect(self.list_double_clicked)
        box.addWidget(self.listview)

        self.addWidget(top)

        self.poly = PolyWidget()
        self.addWidget(self.poly)

    def unselect(self):
        selection_model = self.listview.selectionModel()
        selection_model.select(QModelIndex(), QItemSelectionModel.SelectCurrent)

    def select(self, row):
        selection_model = self.listview.selectionModel()
        index = self.listview.model().index(row, 0, QModelIndex())
        selection_model.select(index, QItemSelectionModel.SelectCurrent)

    def selection_changed(self, selected, deselected):
        try:
            index = selected.indexes()[0]
        except IndexError:
            self.main.set_selected()
            self.poly.show()
            return
        poly = self.db[index.row()]
        self.main.set_selected(poly.lfid)
        self.poly.show(poly)

    def list_double_clicked(self, item):
        self.main.focus(self.db[item.row()].lfid)


class MainWidget(QSplitter):

    def __init__(self, db):
        super().__init__()
        self.db = db
        self.create_ui()

    def create_ui(self):
        self.db_widget = DatabaseWidget(self)

        # Web view
        self.view = QWebEngineView()
        self.view.loadFinished.connect(self.add_polys)

        self.interface = JSInterface(self)
        self.channel = QWebChannel()
        self.channel.registerObject('Interface', self.interface)
        self.view.page().setWebChannel(self.channel)

        html = join(dirname(realpath(__file__)), "assets/map.html")
        self.view.setUrl(QUrl.fromLocalFile(html))

        self.addWidget(self.view)
        self.addWidget(self.db_widget)

    def add_polys(self):
        for poly in self.db:
            points = str(poly.points)
            self.view.page().runJavaScript(f'add_object({points})', poly.set_lfid)

    def set_selected(self, lfid=-1):
        self.view.page().runJavaScript(f'select_object({lfid})')

    def focus(self, lfid):
        self.view.page().runJavaScript(f'focus_object({lfid})')


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
