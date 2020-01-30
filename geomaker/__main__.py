import functools
from contextlib import contextmanager
from os.path import dirname, realpath, join
from operator import attrgetter
import sys
from string import ascii_lowercase
import time

from PyQt5.QtCore import (
    Qt, QObject, QUrl, pyqtSlot, QAbstractListModel, QModelIndex, QVariant, QItemSelectionModel, QThread,
    pyqtSignal
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtWebChannel import QWebChannel
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtWidgets import (
    QApplication, QWidget, QMainWindow, QVBoxLayout, QGridLayout, QListView, QLabel, QInputDialog, QSplitter,
    QFrame, QMessageBox, QComboBox, QPushButton, QTabWidget, QProgressDialog
)

from geomaker.db import PROJECTS, Polygon, Config, db


KEY_MAP = {
    Qt.Key_Space: 'SPC',
    Qt.Key_Escape: 'ESC',
    Qt.Key_Tab: 'TAB',
    Qt.Key_Return: 'RET',
    Qt.Key_Backspace: 'BSP',
    Qt.Key_Delete: 'DEL',
    Qt.Key_Up: 'UP',
    Qt.Key_Down: 'DOWN',
    Qt.Key_Left: 'LEFT',
    Qt.Key_Right: 'RIGHT',
    Qt.Key_Minus: '-',
    Qt.Key_Plus: '+',
    Qt.Key_Equal: '=',
}
KEY_MAP.update({
    getattr(Qt, 'Key_{}'.format(s.upper())): s
    for s in ascii_lowercase
})
KEY_MAP.update({
    getattr(Qt, 'Key_F{}'.format(i)): '<f{}>'.format(i)
    for i in range(1, 13)
})

def key_to_text(event):
    ctrl = event.modifiers() & Qt.ControlModifier
    shift = event.modifiers() & Qt.ShiftModifier

    try:
        text = KEY_MAP[event.key()]
    except KeyError:
        return

    if shift and text.isupper():
        text = 'S-{}'.format(text)
    elif shift:
        text = text.upper()
    if ctrl:
        text = 'C-{}'.format(text)

    return text


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

    def before_reset(self, lfid):
        interface.select_poly(-1)
        self._selected = lfid
        self.beginResetModel()

    def after_reset(self):
        self.endResetModel()
        interface.select_poly(self._selected)

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
        db.update_points(lfid, data)
        db_widget.select(db.index(lfid=lfid))

    @pyqtSlot(int)
    def remove_poly(self, lfid):
        db.delete(lfid)

    @pyqtSlot(int)
    def select_poly(self, lfid=-1):
        if lfid < 0:
            db_widget.unselect()
        else:
            db_widget.select(db.index(lfid=lfid))

    @pyqtSlot(int)
    def open_poly(self, lfid):
        main_widget.focus(lfid)

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
        self._add_row(combobox, attrname='project_chooser')
        self._add_row(label(''), title=label('Dedicated'), attrname='dedicated')
        nrows_b = self._add_row(label(''), title=label('Tiles'), attrname='tiles')

        self.dedicated.setTextInteractionFlags(Qt.TextBrowserInteraction)
        self.dedicated.linkActivated.connect(self.dl_dedicated)
        self.tiles.setTextInteractionFlags(Qt.TextBrowserInteraction)
        self.tiles.linkActivated.connect(self.dl_tiles)

        self._add_row(label('Bomgobob'), attrname='image')

        self.box.addWidget(frame(QFrame.VLine), nrows_a + 2, 1, nrows_b - nrows_a - 2, 1)
        self.box.setRowStretch(nrows_b + 1, 1)

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

    @property
    def project(self):
        return PROJECTS[self.project_chooser.currentIndex()][0]

    def update_project(self):
        if self.poly.dedicated(self.project):
            text = 'Yes'
        elif self.poly.job(self.project, True):
            job = self.poly.job(self.project, True)
            text = f'Exporting ({job.stage})'
        else:
            text = 'No'
        self.dedicated.setText(f'{text} (<a href="dl">download</a>)')

        ntiles = self.poly.ntiles(self.project)
        if ntiles > 0:
            text = str(ntiles)
        elif self.poly.job(self.project, False):
            job = self.poly.job(self.project, False)
            text = f'Exporting ({job.stage})'
        else:
            text = 'No'
        self.tiles.setText(f'{text} (<a href="dl">download</a>)')

        thumb = self.poly.thumbnail(self.project)
        if thumb:
            pixmap = QPixmap(thumb.filename)
            w = self.image.width()
            h = max(w, int(w / pixmap.width() * pixmap.height()))
            self.image.setPixmap(pixmap.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            self.image.setPixmap(QPixmap())

    def _create_job(self, dedicated):
        if self.poly.job(self.project, dedicated):
            answer = QMessageBox.question(
                self, 'Delete existing job?',
                'This region already has a job of this type. Delete it and restart?'
            )
            if answer == QMessageBox.No:
                return
            self.poly.delete_job(dedicated)

        error = self.poly.create_job(self.project, dedicated, email=config['email'])
        if error:
            QMessageBox().critical(self, 'Error', error)

        self.update_project()

    def dl_dedicated(self):
        if self.poly.dedicated(self.project):
            answer = QMessageBox.question(
                self, 'Delete existing dedicated file?',
                'This region already has a dedicated data file. Delete it and download again?'
            )
            if answer == QMessageBox.No:
                return
            self.poly.delete_dedicated(self.project)
        self._create_job(dedicated=True)

    def dl_tiles(self):
        if self.poly.ntiles(self.project) > 0:
            answer = QMessageBox.question(
                self, 'Disassociate existing tiles?',
                'This region already has associated tiles. Disassociate them and download again?'
            )
            if answer == QMessageBox.No:
                return
            self.poly.delete_tiles(self.project)
        self._create_job(dedicated=False)


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

        self.tabview = QTabWidget()
        self.addWidget(self.tabview)

        self.tabview.addTab(db_widget, 'Regions')

    def add_polys(self):
        for poly in db:
            points = str(list(poly.geometry))
            self.view.page().runJavaScript(
                f'add_object({points})',
                functools.partial(Polygon.lfid.fset, poly)
            )

    def set_selected(self, lfid=-1):
        self.view.page().runJavaScript(f'select_object({lfid})')

    def focus(self, lfid):
        self.view.page().runJavaScript(f'focus_object({lfid})')


def progress(items, desc, length=None):
    if length is None:
        length = len(items)
    if length == 0:
        return

    progress = QProgressDialog(desc, 'Cancel', 0, length, main_window)
    progress.setValue(0)
    progress.setWindowTitle('GeoMaker')
    progress.setWindowModality(Qt.WindowModal)
    progress.setMinimumDuration(0)
    progress.show()

    # This is incredibly hacky, but seems to be necessary
    # to get the progress dialog to draw at 0%
    # TODO: Figure out a way to get rid of this
    for _ in range(10):
        app.processEvents()
        time.sleep(0.01)

    for i, item in enumerate(items):
        app.processEvents()
        progress.setValue(i)
        if progress.wasCanceled():
            return
        yield item
    else:
        progress.setValue(progress.maximum())


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

    def keyPressEvent(self, event):
        text = key_to_text(event)
        if text == 'DEL':
            print('delete')
        elif text == '<f5>':
            with db.session() as s:
                for job in progress(db.jobs(), 'Refreshing jobs...', length=db.njobs()):
                    job.refresh()
                for job in progress(db.jobs(stage='complete'), 'Downloading data...', length=db.njobs(stage='complete')):
                    job.download()
            db_widget.poly.update_project()


def main():
    global app, config, db_widget, interface, main_widget, main_window

    config = Config()

    app = QApplication(sys.argv)
    db_widget = DatabaseWidget()
    interface = JSInterface()
    main_widget = MainWidget()
    main_window = MainWindow()

    main_window.showMaximized()
    sys.exit(app.exec_())
