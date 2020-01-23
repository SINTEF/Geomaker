import functools
from os.path import dirname, realpath, join
from operator import attrgetter
import sys

from PyQt5.QtCore import (
    Qt, QObject, QUrl, pyqtSlot, QAbstractListModel, QModelIndex, QVariant, QItemSelectionModel
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtWebChannel import QWebChannel
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtWidgets import (
    QApplication, QWidget, QMainWindow, QVBoxLayout, QGridLayout, QListView, QLabel, QInputDialog, QSplitter,
    QFrame, QMessageBox, QComboBox, QPushButton
)

from geomaker.db import PROJECTS, Database, Polygon, Config, Status


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

    def before_reset(self, poly):
        interface.select_poly(-1)
        self._selected = poly
        self.beginResetModel()

    def after_reset(self):
        self.endResetModel()
        interface.select_poly(self._selected.lfid)

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

    @pyqtSlot(str)
    def print(self, s):
        print(s)


class PolyWidget(QWidget):

    def __init__(self):
        super().__init__()
        self.poly = None
        self.setVisible(False)
        self.create_ui()

    def _add_row(self, widget, title=None, attrname=None, align=None):
        if not hasattr(self, '_rows'):
            self._rows = 0
        if title is None:
            if align is not None:
                self.box.addWidget(widget, self._rows, 0, 1, 3, align)
            else:
                self.box.addWidget(widget, self._rows, 0, 1, 3)
        else:
            self.box.addWidget(title, self._rows, 0, Qt.AlignRight)
            self.box.addWidget(widget, self._rows, 2, Qt.AlignLeft)
        if attrname is not None:
            setattr(self, attrname, widget)
        self._rows += 1
        return self._rows

    def create_ui(self):
        self.box = QGridLayout()
        self.setLayout(self.box)

        self._add_row(label(''), attrname='name', align=Qt.AlignCenter)
        self._add_row(frame(QFrame.HLine))
        self._add_row(label(''), title=label('West'), attrname='west')
        self._add_row(label(''), title=label('East'), attrname='east')
        self._add_row(label(''), title=label('South'), attrname='south')
        self._add_row(label(''), title=label('North'), attrname='north')
        nrows_a = self._add_row(label(''), title=label('Area'), attrname='area')
        self.box.addWidget(frame(QFrame.VLine), 2, 1, nrows_a - 2, 1)

        self._add_row(frame(QFrame.HLine))

        combobox = QComboBox()
        combobox.addItems(project for _, project in PROJECTS)
        combobox.currentIndexChanged.connect(self.update_project)
        self._add_row(combobox, attrname='project')
        nrows_b = self._add_row(label(''), title=label('Status'), attrname='status')

        button = QPushButton('')
        button.clicked.connect(self.act)
        self._add_row(button, attrname='action', align=Qt.AlignCenter)

        self._add_row(label(''), attrname='image')

        self.box.addWidget(frame(QFrame.VLine), nrows_b - 1, 1, nrows_b - nrows_a - 2, 1)
        self.box.setRowStretch(nrows_b + 2, 1)

    def show(self, poly=None):
        self.poly = poly

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

        self.update_project()

    def update_project(self):
        if self.poly is None:
            return
        project, _ = PROJECTS[self.project.currentIndex()]
        status = self.poly.status(project)

        self.status.setText(status.desc())
        self.action.setText(status.action())

        if status == Status.Downloaded:
            pixmap = QPixmap()
            pixmap.loadFromData(self.poly.geotiff(project).thumbnail())
            w = self.image.width()
            h = max(w, int(w / pixmap.width() * pixmap.height()))
            self.image.setPixmap(pixmap.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            self.image.setPixmap(QPixmap())

    def act(self):
        if self.poly is None:
            return
        project, _ = PROJECTS[self.project.currentIndex()]
        status = self.poly.status(project)

        if status == Status.Nothing or status == Status.ExportErrored:
            self.poly.export(project, config['email'])
        elif status == Status.ExportWaiting or status == Status.ExportProcessing:
            self.poly.refresh(project)
        elif status == Status.DownloadReady:
            self.poly.download(project)

        self.update_project()
        if self.poly.status(project) == Status.ExportErrored:
            QMessageBox.critical(self, 'Error', self.poly.error(project))


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
        self.listview.setEditTriggers(QListView.EditKeyPressed | QListView.SelectedClicked)
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

        self.channel = QWebChannel()
        self.channel.registerObject('Interface', interface)
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

    def showMaximized(self):
        super().showMaximized()
        config.verify(self)

    def message(self, title, msg):
        QMessageBox.information(self, title, msg)

    def query_str(self, title, msg):
        name, _ = QInputDialog.getText(self, title, msg)
        return name


def main():
    global config, db, db_widget, interface, main_widget

    config = Config()

    app = QApplication(sys.argv)
    db = Database()
    db_widget = DatabaseWidget()
    interface = JSInterface()
    main_widget = MainWidget()

    win = MainWindow()
    win.showMaximized()
    sys.exit(app.exec_())
