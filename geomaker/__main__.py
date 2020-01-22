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
    QApplication, QWidget, QMainWindow, QVBoxLayout, QGridLayout, QListView, QLabel, QInputDialog, QSplitter,
    QFrame
)

from geomaker.db import Database, Polygon


def label(text):
    label = QLabel()
    label.setText(text)
    return label


def frame(orientation):
    frame = QFrame()
    frame.setFrameShape(orientation)
    return frame


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
            return QVariant(db[index.row()].name)
        return QVariant()

    def rowCount(self, parent):
        return len(db)


class JSInterface(QObject):

    def __init__(self):
        super().__init__()

    @pyqtSlot(int, str)
    def add_poly(self, lfid, data):
        name, accept = QInputDialog.getText(main_widget, 'Name', 'Name this region:')
        if accept:
            db.create(lfid, name, data)
            self.select_poly()
            self.select_poly(lfid)

    @pyqtSlot(int, str)
    def edit_poly(self, lfid, data):
        db.update(lfid, data)
        db_widget.select(db.index_of(lfid=lfid))

    @pyqtSlot(int)
    def remove_poly(self, lfid):
        db.delete(lfid)

    @pyqtSlot(int)
    def select_poly(self, lfid=-1):
        if lfid < 0:
            db_widget.unselect()
        else:
            db_widget.select(db.index_of(lfid=lfid))


class PolyWidget(QWidget):

    def __init__(self):
        super().__init__()
        self.setVisible(False)
        self.create_ui()

    def _add_row(self, attrname, title, widget):
        self.box.addWidget(label(title), self._rows, 0, Qt.AlignRight)
        self.box.addWidget(widget, self._rows, 2, Qt.AlignLeft)
        setattr(self, attrname, widget)
        self._rows += 1

    def create_ui(self):
        self.box = QGridLayout()
        self.setLayout(self.box)

        self.name = label('')
        self.box.addWidget(self.name, 0, 0, 1, 3, Qt.AlignCenter)
        self.box.addWidget(frame(QFrame.HLine), 1, 0, 1, 3)

        self._rows = 2
        self._add_row('west', 'West', label(''))
        self._add_row('east', 'East', label(''))
        self._add_row('south', 'South', label(''))
        self._add_row('north', 'North', label(''))
        self._add_row('area', 'Area', label(''))

        self.box.addWidget(frame(QFrame.VLine), 2, 1, self._rows - 2, 1)
        self.box.setRowStretch(self._rows, 1)

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
        self.area.setText(f'{poly.area/1000000:.3f} km<sup>2</sup>')


class DatabaseWidget(QSplitter):

    def __init__(self):
        super().__init__()
        self.setOrientation(Qt.Vertical)
        self.create_ui()

    def create_ui(self):
        top = QWidget()
        box = QVBoxLayout()
        top.setLayout(box)

        box.addWidget(label('<strong>Stored regions</strong>'))

        self.listview = QListView()
        self.listview.setModel(DatabaseModel(db))
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
            main_widget.set_selected()
            self.poly.show()
            return
        poly = db[index.row()]
        main_widget.set_selected(poly.lfid)
        self.poly.show(poly)

    def list_double_clicked(self, item):
        main_widget.focus(db[item.row()].lfid)


class MainWidget(QSplitter):

    def __init__(self):
        super().__init__()
        self.create_ui()

    def create_ui(self):
        self.view = QWebEngineView()
        self.view.loadFinished.connect(self.add_polys)

        self.interface = JSInterface()
        self.channel = QWebChannel()
        self.channel.registerObject('Interface', self.interface)
        self.view.page().setWebChannel(self.channel)

        html = join(dirname(realpath(__file__)), "assets/map.html")
        self.view.setUrl(QUrl.fromLocalFile(html))

        self.addWidget(self.view)
        self.addWidget(db_widget)

    def add_polys(self):
        for poly in db:
            points = str(poly.points)
            self.view.page().runJavaScript(f'add_object({points})', poly.set_lfid)

    def set_selected(self, lfid=-1):
        self.view.page().runJavaScript(f'select_object({lfid})')

    def focus(self, lfid):
        self.view.page().runJavaScript(f'focus_object({lfid})')


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle('GeoMaker')
        self.setCentralWidget(main_widget)


def main():
    global db, db_widget, main_widget

    app = QApplication(sys.argv)
    db = Database()
    db_widget = DatabaseWidget()
    main_widget = MainWidget()

    win = MainWindow()
    win.showMaximized()
    sys.exit(app.exec_())
