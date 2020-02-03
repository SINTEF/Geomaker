import functools
from contextlib import contextmanager
from os.path import dirname, realpath, join
from operator import attrgetter
import sys
import time

from PyQt5.QtCore import (
    Qt, QObject, QAbstractListModel, QEvent, QItemSelectionModel,
    QModelIndex, QThread, QUrl, QVariant, pyqtSignal, pyqtSlot,
)
from PyQt5.QtGui import QPixmap, QIcon, QKeyEvent
from PyQt5.QtWebChannel import QWebChannel
from PyQt5.QtWidgets import (
    QApplication, QDialog, QInputDialog, QMainWindow, QMessageBox, QProgressBar, QPushButton, QWidget,
)

from .ui.utils import key_to_text, angle_to_degrees
from .db import PROJECTS, Polygon, Config, db


class ProjectsModel(QAbstractListModel):

    def rowCount(self, parent):
        return len(PROJECTS)

    def data(self, index, role):
        if role == Qt.DisplayRole:
            return QVariant(PROJECTS[index.row()][1])
        return QVariant()


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

    # Emitted whenever a new polygon is edited by leaflet
    polygon_added = pyqtSignal(int, str)

    # Emitted whenever a polygon is selected in leaflet
    polygon_selected = pyqtSignal(int)

    # Emitted whenever a polygon is double-clicked in leaflet
    polygon_double_clicked = pyqtSignal(int)

    # Emitted whenever a new polygon is edited by leaflet
    polygon_edited = pyqtSignal(int, str)

    # Emitted whenever a polygon was deleted by leaflet
    polygon_deleted = pyqtSignal(int)

    # Every possible signal signature (with an added str in front)
    # must be listed here.  This function is the only point of
    # interaction visible from Javascript for emitting events.
    @pyqtSlot(str, int)
    @pyqtSlot(str, int, str)
    def emit(self, name, *args):
        getattr(self, name).emit(*args)


from geomaker.ui.interface import Ui_MainWindow
from geomaker.ui.thumbnail import Ui_Thumbnail
from geomaker.ui.jobdialog import Ui_JobDialog


class JobDialog(QDialog):

    def __init__(self, poly, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.poly = poly

        self.ui = Ui_JobDialog()
        self.ui.setupUi(self)

        self.ui.email.setText(config['email'])
        self.ui.projectlist.setModel(ProjectsModel())

    def done(self, result):
        project, _ = PROJECTS[self.ui.projectlist.selectedIndexes()[0].row()]
        dedicated = self.ui.dedicated.checkState() == Qt.Checked

        # Set these attributes so that the caller can access them
        self.project = project
        self.dedicated = dedicated

        if result != QDialog.Accepted:
            return super().done(result)

        if dedicated and self.poly.dedicated(project):
            answer = QMessageBox.question(
                self, 'Delete existing dedicated file?',
                'This region already has a dedicated data file. Delete it and download again?'
            )
            if answer == QMessageBox.No:
                return
            self.poly.delete_dedicated(project)

        if not dedicated and self.poly.ntiles(project) > 0:
            answer = QMessageBox.question(
                self, 'Disassociate existing tiles?',
                'This region already has associated tiles. Disassociate them and download again?'
            )
            if answer == QMessageBox.No:
                return
            self.poly.delete_tiles(project)

        if self.poly.job(project, dedicated):
            answer = QMessageBox.question(
                self, 'Delete existing job?',
                'This region already has a job of this type. Delete it and restart?'
            )
            if answer == QMessageBox.No:
                return
            self.poly.delete_job(project, dedicated)

        self.poly.create_job(project, dedicated, self.ui.email.text())
        return super().done(QDialog.Accepted)


class ThumbnailWidget(QWidget):

    def __init__(self, project, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._project = project

        self.ui = Ui_Thumbnail()
        self.ui.setupUi(self)

    def update_poly(self, poly):
        thumb = poly.thumbnail(self._project)
        if thumb is not None:
            pixmap = QPixmap(thumb.filename)
            self.ui.thumbnail.setPixmap(pixmap.scaledToWidth(
                self.ui.thumbnail.width(), Qt.SmoothTransformation
            ))
            return
        self.ui.thumbnail.setPixmap(QPixmap())
        if poly.njobs(project=self._project) > 0:
            self.ui.thumbnail.setText(f'A job for {self._project} is currently running')


class KeyFilter(QObject):

    def __init__(self, ui):
        super().__init__()
        self.ui = ui

    def eventFilter(self, obj, event):
        if not isinstance(event, QKeyEvent) or event.type() != QEvent.KeyPress:
            return super().eventFilter(obj, event)
        text = key_to_text(event)
        if text == '<f5>':
            self.ui.update_jobs()
        else:
            return super().eventFilter(obj, event)
        return True


class GUI(Ui_MainWindow):

    def __init__(self):
        super().__init__()

        self._poly = None
        self._project_tabs = {}

        self._thread = None
        self._worker = None
        self._continue = None

        self.main = QMainWindow()
        self.setupUi(self.main)
        self.main.showMaximized()

    def setupUi(self, main):
        super().setupUi(main)
        self.polydetails.hide()

        # The JS-Python interface allows us to capture events from JS
        self.js_interface = JSInterface()
        self.js_interface.polygon_added.connect(self.webview_polygon_added)
        self.js_interface.polygon_selected.connect(self.webview_selection_changed)
        self.js_interface.polygon_double_clicked.connect(self.webview_double_clicked)
        self.js_interface.polygon_edited.connect(self.webview_polygon_edited)
        self.js_interface.polygon_deleted.connect(self.webview_polygon_deleted)

        # Web view that exposes the interface to JS via a QWebChannel
        self.channel = QWebChannel()
        self.channel.registerObject('Interface', self.js_interface)
        self.webview.page().setWebChannel(self.channel)
        html = join(dirname(realpath(__file__)), "assets/map.html")
        self.webview.setUrl(QUrl.fromLocalFile(html))
        self.webview.loadFinished.connect(self.webview_finished_load)

        # Polygon list item model and events
        self.polylist.setModel(DatabaseModel(db))
        self.polylist.selectionModel().selectionChanged.connect(self.polylist_selection_changed)
        self.polylist.doubleClicked.connect(self.polylist_double_clicked)

        # Create thumbnail widgets for each project
        for project, _ in PROJECTS:
            self._project_tabs[project] = ThumbnailWidget(project)

        # New job button
        self.downloadbtn = QPushButton()
        icon = QIcon()
        icon.addPixmap(QPixmap(':/icons/download.png'), QIcon.Normal, QIcon.On)
        self.downloadbtn.setIcon(icon)
        self.projects.setCornerWidget(self.downloadbtn, Qt.TopRightCorner)
        self.downloadbtn.clicked.connect(self.start_new_job)

        # Keys
        self.keyfilter = KeyFilter(self)
        main.installEventFilter(self.keyfilter)

        # Progress in status bar
        self.progress = QProgressBar()
        self.progress.setMinimum(0)
        self.statusbar.addWidget(self.progress, 1)

    @property
    def poly(self):
        return self._poly

    @poly.setter
    def poly(self, poly):
        self._poly = poly
        if poly is None:
            self.polydetails.hide()
            return
        self.polydetails.show()

        self.polydetails.setTitle(poly.name)
        self.west.setText(f'{poly.west:.4f} ({angle_to_degrees(poly.west, "WE")})')
        self.east.setText(f'{poly.east:.4f} ({angle_to_degrees(poly.east, "WE")})')
        self.south.setText(f'{poly.south:.4f} ({angle_to_degrees(poly.south, "SN")})')
        self.north.setText(f'{poly.north:.4f} ({angle_to_degrees(poly.north, "SN")})')
        self.area.setText(f'{poly.area/1000000:.3f} km<sup>2</sup>')

        # Remove existing project tabs and add new ones
        # Attempt to keep the current tab on the same project if possible
        selected_widget = self.projects.currentWidget()
        while self.projects.count() > 0:
            self.projects.removeTab(0)
        for project, _ in PROJECTS:
            self.refresh_tabs_hint(project, select=False)
            self._project_tabs[project].update_poly(poly)
        new_index = max(0, self.projects.indexOf(selected_widget))
        self.projects.setCurrentIndex(new_index)

    def refresh_tabs_hint(self, project, select=True):
        activate = self.poly.thumbnail(project) or self.poly.njobs(project=project) > 0
        widget = self._project_tabs[project]
        index = self.projects.indexOf(widget)

        # The page activation state is up to date
        if activate == index > -1:
            if activate and select:
                self.projects.setCurrentIndex(index)
            return

        # The page must be removed
        if not activate:
            self.projects.removeTab(index)
            self.projects.setCurrentIndex(0)
            return

        # The page must be added
        insertion_index = sum(
            1 for proj, page in self._project_tabs.items()
            if proj < project and self.projects.indexOf(page) > -1
        )
        self.projects.insertTab(insertion_index, widget, project)
        if select:
            self.projects.setCurrentIndex(insertion_index)

    def js(self, code, callback=None):
        if callback is None:
            self.webview.page().runJavaScript(code)
        else:
            self.webview.page().runJavaScript(code, callback)

    def webview_finished_load(self):
        for poly in db:
            points = str(list(poly.geometry))
            self.js(f'add_object({points})', functools.partial(Polygon.lfid.fset, poly))
        self.webview.page().runJavaScript('focus_object(-1)')

    def webview_selection_changed(self, lfid):
        if lfid < 0:
            index = QModelIndex()
        else:
            index = self.polylist.model().index(db.index(lfid=lfid), 0, QModelIndex())
        selection_model = self.polylist.selectionModel()
        selection_model.select(index, QItemSelectionModel.SelectCurrent)

    def webview_double_clicked(self, lfid):
        self.js(f'focus_object({lfid})')

    def webview_polygon_added(self, lfid, data):
        name, accept = QInputDialog.getText(self.main, 'Name', 'Name this region:')
        if accept:
            db.create(lfid, name, data)
            self.webview_selection_changed(lfid)

    def webview_polygon_edited(self, lfid, data):
        db.update_points(lfid, data)
        self.webview_selection_changed(lfid)

    def webview_polygon_deleted(self, lfid):
        db.delete(lfid)

    def polylist_selection_changed(self, selected, deselected):
        try:
            index = selected.indexes()[0]
        except IndexError:
            self.js('select_object(-1)')
            self.poly = None
            return
        poly = db[index.row()]
        self.js(f'select_object({poly.lfid})')
        self.poly = poly

    def polylist_double_clicked(self, item):
        poly = db[item.row()]
        self.js(f'focus_object({poly.lfid})')

    def project_tab_clicked(self, index):
        if index == self.projects.count() - 1:
            self.start_new_job()

    def start_new_job(self):
        if self.poly is None:
            return
        dialog = JobDialog(self.poly)
        retval = dialog.exec_()
        if retval == QDialog.Accepted:
            self.refresh_tabs_hint(dialog.project)

    def run_thread(self, jobs, text, cont=None):
        assert not self._thread
        if len(jobs) == 0:
            if cont is not None:
                cont()
            return

        thread = QThread()
        worker = Worker(jobs)
        worker.moveToThread(thread)
        thread.started.connect(worker.process)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._thread_finished)
        worker.callback.connect(self._thread_callback)

        self._thread = thread
        self._worker = worker
        self._continue = cont

        self.progress.setMaximum(len(jobs))
        self.progress.setValue(0)
        self.progress.setFormat(f'{text} (%v/{len(jobs)} · %p%)')

        thread.start()

    def _thread_callback(self, result):
        poly, project = result()
        if poly is self.poly:
            self._project_tabs[project].update_poly(poly)
            self.refresh_tabs_hint(project, select=False)
        self.progress.setValue(self.progress.value() + 1)

    def _thread_finished(self):
        self._worker = None
        self._thread = None
        self.progress.setValue(0)
        self.progress.setMaximum(1)
        self.progress.setFormat('')
        if self._continue is not None:
            self._continue()

    def update_jobs(self):
        self.run_thread(
            [job.refresh(async=True) for job in db.jobs()],
            'Refreshing jobs', cont=self.download_jobs,
        )

    def download_jobs(self):
        self.run_thread(
            [job.download(async=True) for job in db.jobs(stage='complete')],
            'Downloading jobs',
        )


class Worker(QObject):

    finished = pyqtSignal()
    callback = pyqtSignal(object)
    progress = pyqtSignal(int, int)

    def __init__(self, workers):
        super().__init__()
        self.workers = workers

    @pyqtSlot()
    def process(self):
        for worker in self.workers:
            self.callback.emit(worker())
        self.finished.emit()


def main():
    global config
    config = Config()

    app = QApplication(sys.argv)
    gui = GUI()
    sys.exit(app.exec_())
