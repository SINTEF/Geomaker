from contextlib import contextmanager
from functools import partial
from os.path import dirname, realpath, join
from operator import attrgetter
from pathlib import Path
import sys
import time

import numpy as np
from splipy import Surface, BSplineBasis
from splipy.io import G2
from stl.mesh import Mesh as STLMesh

from PyQt5.QtCore import (
    Qt, QObject, QEvent, QItemSelectionModel, QModelIndex, QThread,
    QUrl, pyqtSignal, pyqtSlot,
)
from PyQt5.QtGui import QPixmap, QIcon, QKeyEvent
from PyQt5.QtWebChannel import QWebChannel
from PyQt5.QtWidgets import (
    QApplication, QDialog, QFileDialog, QInputDialog, QLayout,
    QMainWindow, QMessageBox, QProgressBar, QPushButton, QWidget,
)

from .ui.utils import key_to_text, angle_to_degrees
from .ui.models import ProjectsModel, DatabaseModel
from .db import PROJECTS, Polygon, ConfigFile, DataFile, db
from . import image

# Classes generated by qtdesigner
from .ui.interface import Ui_MainWindow
from .ui.thumbnail import Ui_Thumbnail
from .ui.jobdialog import Ui_JobDialog
from .ui.exporter import Ui_Exporter


@contextmanager
def block_signals(obj):
    prev = obj.blockSignals(True)
    yield obj
    obj.blockSignals(prev)


class JSInterface(QObject):
    """Class that marshals JS events to Python.
    This is a Qt object that can emit signals upon calls from JS.
    """

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


class ExporterDialog(QDialog):
    """A dialog window for exporting data."""

    IMAGE_FORMATS = {'png', 'jpeg'}

    FORMATS = [
        ('png', {'png'}, 'Portable Network Graphics (PNG)'),
        ('jpeg', {'jpg', 'jpeg'}, 'Joint Photographic Experts Group (JPEG)'),
        ('g2', {'g2'}, 'GoTools B-Splines (G2)'),
        ('stl', {'stl'}, 'Stereolithography (STL)'),
    ]

    COORDS = [
        ('utm33n', 'UTM Zone 33 North', 'm'),
        ('latlon', 'Latitude and Longitude', '°'),
    ]

    def __init__(self, poly, project, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.poly = poly
        self.project = project
        self.selected_filter = None
        self.ui = Ui_Exporter()
        self.ui.setupUi(self)

        # Populate dropdown boxes
        self.ui.formats.addItems([name for _, _, name in self.FORMATS])
        self.ui.coordinates.addItems([name for _, name, _ in self.COORDS])
        self.ui.colormaps.addItems(sorted(image.list_colormaps()))

        # Set the default settings based on the previous time
        data = DataFile()
        self.format = data.get('export-format', 'png')
        self.ui.filename.addItems(data.get('export-filenames', []))
        self.ui.resolution.setValue(data.get('export-resolution', 1.0))
        self.ui.zero_sea_level.setChecked(data.get('export-zero-sea', True))
        self.boundary_mode = data.get('export-boundary-mode', 'interior')
        self.rotation_mode = data.get('export-rotation-mode', 'north')
        self.coords = data.get('export-coords', 'utm33n')
        self.colormap = data.get('export-colormap', 'terrain')

        if self.ui.filename.currentText().strip() == '':
            self.ui.filename.setEditText(poly.name)

        # Connect signals
        self.ui.filename.currentTextChanged.connect(self.filename_changed)
        self.ui.browsebtn.clicked.connect(self.browse)
        self.ui.formats.currentIndexChanged.connect(self.format_changed)
        self.ui.coordinates.currentIndexChanged.connect(self.coords_changed)
        self.ui.okbtn.clicked.connect(self.accept)
        self.ui.cancelbtn.clicked.connect(self.reject)
        self.ui.refreshbtn.clicked.connect(self.recompute)

        # Trigger some basic validation
        self.format_changed()
        self.update_resolution_suffix()

        self.setFixedSize(self.size())

    @property
    def boundary_mode(self):
        if self.ui.interior_bnd.isChecked():
            return 'interior'
        return 'exterior'

    @boundary_mode.setter
    def boundary_mode(self, value):
        if value == 'interior':
            self.ui.interior_bnd.setChecked(True)
        else:
            self.ui.exterior_bnd.setChecked(True)

    @property
    def rotation_mode(self):
        if self.ui.no_rot.isChecked():
            return 'none'
        elif self.ui.north_rot.isChecked():
            return 'north'
        return 'free'

    @rotation_mode.setter
    def rotation_mode(self, value):
        if value == 'none':
            self.ui.no_rot.setChecked(True)
        elif value == 'north':
            self.ui.north_rot.setChecked(True)
        else:
            self.ui.free_rot.setChecked(True)

    @property
    def coords(self):
        return self.COORDS[self.ui.coordinates.currentIndex()][0]

    @coords.setter
    def coords(self, value):
        self.ui.coordinates.setCurrentIndex(next(i for i, (v, _, _) in enumerate(self.COORDS) if v == value))

    @property
    def coords_unit(self):
        return self.COORDS[self.ui.coordinates.currentIndex()][2]

    @property
    def colormap(self):
        return self.ui.colormaps.currentText()

    @colormap.setter
    def colormap(self, value):
        self.ui.colormaps.setCurrentText(value)

    @property
    def format(self):
        return self.FORMATS[self.ui.formats.currentIndex()][0]

    @format.setter
    def format(self, value):
        self.ui.formats.setCurrentIndex(next(i for i, (fmt, _, _) in enumerate(self.FORMATS) if value == fmt))

    @property
    def format_suffixes(self):
        return self.FORMATS[self.ui.formats.currentIndex()][1]

    @property
    def image_mode(self):
        return self.format in self.IMAGE_FORMATS

    def format_changed(self):
        self.ui.exterior_bnd.setEnabled(self.image_mode)
        self.ui.interior_bnd.setEnabled(self.image_mode)
        self.ui.no_rot.setEnabled(self.image_mode)
        self.ui.north_rot.setEnabled(self.image_mode)
        self.ui.free_rot.setEnabled(self.image_mode)
        self.ui.colormaps.setEnabled(self.image_mode)

        filename = Path(self.ui.filename.currentText())
        if filename.suffix[1:] not in self.format_suffixes:
            with block_signals(self.ui.filename) as obj:
                obj.setEditText(str(filename.with_suffix('.' + self.format)))

    def coords_changed(self):
        # Very basic conversion between degrees and meters:
        # originally, a meter was defined as one 10000th of the
        # distance between the equator and the pole, so a degree of
        # latitude is roughly 10000/90 meters.
        if self.coords_unit == '°' and self.ui.resolution.suffix() == 'm':
            self.ui.resolution.setValue(self.ui.resolution.value() / 10000 * 90)
        elif self.coords_unit == 'm' and self.ui.resolution.suffix() == '°':
            self.ui.resolution.setValue(self.ui.resolution.value() / 90 * 10000)
        self.update_resolution_suffix()

    def update_resolution_suffix(self):
        self.ui.resolution.setSuffix(self.coords_unit)

    def recompute(self):
        if self.boundary_mode == 'actual':
            self.ui.fitwarning.setText('')
            return
        pctg, theta = self.poly.check_area(self.boundary_mode, self.rotation_mode, self.coords)
        if self.boundary_mode == 'interior':
            self.ui.fitwarning.setText(f'{100*pctg:.2f}% shortfall, rotated {theta*180/np.pi:.2f}°')
        else:
            self.ui.fitwarning.setText(f'{100*pctg:.2f}% overshoot, rotated {theta*180/np.pi:.2f}°')

    def browse(self):
        filters = [
            'Images (*.png *.jpg *.jpeg)',
            'GoTools (*.g2)',
            'Stereolithography (*.stl)'
        ]
        filename, selected_filter = QFileDialog.getSaveFileName(
            self, 'Save file', self.ui.filename.currentText(),
            ';;'.join(filters), DataFile().get('export-filter', filters[0]),
        )

        if filename:
            self.ui.filename.setEditText(filename)
            self.selected_filter = selected_filter

    def filename_changed(self):
        suffix = Path(self.ui.filename.currentText()).suffix[1:]
        if suffix is None:
            return
        try:
            format_index = next(i for i, (_, fmts, _) in enumerate(self.FORMATS) if suffix in fmts)
            self.ui.formats.setCurrentIndex(format_index)
        except StopIteration:
            pass

    def update_datafile(self):
        """Update persisted data so that the export settings won't
        have to be changed so often.
        """
        filename = self.ui.filename.currentText()

        with DataFile().suspend_write() as data:
            if filename:
                past_filenames = data.get('export-filenames', [])
                try:
                    past_filenames.remove(filename)
                except ValueError:
                    pass
                past_filenames.insert(0, filename)
                data['export-filenames'] = past_filenames[:100]

            if self.selected_filter:
                data['export-filter'] = self.selected_filter

            data.update({
                'export-format': self.format,
                'export-resolution': self.ui.resolution.value(),
                'export-coords': self.coords,
                'export-zero-sea': self.ui.zero_sea_level.isChecked(),
            })

            if self.image_mode:
                data.update({
                    'export-colormap': self.colormap,
                    'export-boundary-mode': self.boundary_mode,
                    'export-rotation-mode': self.rotation_mode,
                })

    def done(self, result):
        if result != QDialog.Accepted:
            return super().done(result)
        self.update_datafile()

        if self.image_mode:
            boundary_mode = self.boundary_mode
            rotation_mode = self.rotation_mode
        else:
            boundary_mode = 'actual'
            rotation_mode = 'none'

        if self.format == 'stl':
            x, y, tri = self.poly.generate_triangulation(self.coords, self.ui.resolution.value())
        else:
            x, y = self.poly.generate_meshgrid(
                boundary_mode, rotation_mode,
                self.coords, self.ui.resolution.value()
            )

        if self.image_mode:
            data = self.poly.interpolate(self.project, x, y)
        else:
            data = self.poly.interpolate(self.project, y, x)
        if not self.ui.zero_sea_level.isChecked():
            data -= np.min(data)

        if self.image_mode:
            image.array_to_image(data, self.format, self.colormap, self.ui.filename.currentText())
        elif self.format == 'g2':
            cpts = np.stack([x, y, data], axis=2)
            knots = [[0.0] + list(map(float, range(n))) + [float(n-1)] for n in data.shape]
            bases = [BSplineBasis(order=2, knots=kts) for kts in knots]
            srf = Surface(*bases, cpts, raw=True)
            with G2(self.ui.filename.currentText()) as g2:
                g2.write(srf)
        elif self.format == 'stl':
            mesh = STLMesh(np.zeros(tri.shape[0], STLMesh.dtype))
            mesh.vectors[:,:,0] = x[tri]
            mesh.vectors[:,:,1] = y[tri]
            mesh.vectors[:,:,2] = data[tri]
            mesh.save(self.ui.filename.currentText())

        return super().done(QDialog.Accepted)


class JobDialog(QDialog):
    """A dialog window for starting a new job."""

    def __init__(self, poly, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.poly = poly

        self.ui = Ui_JobDialog()
        self.ui.setupUi(self)

        self.ui.email.setText(ConfigFile()['email'])
        self.ui.projectlist.setModel(ProjectsModel())

        self.ui.okbtn.clicked.connect(self.accept)
        self.ui.cancelbtn.clicked.connect(self.reject)

        self.setFixedSize(self.size())

    def done(self, result):
        project, _ = PROJECTS[self.ui.projectlist.selectedIndexes()[0].row()]
        dedicated = self.ui.dedicated.checkState() == Qt.Checked

        # Set these attributes so that the caller can access them
        self.project = project
        self.dedicated = dedicated

        # If the dialog was canceled, no further validation
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

        retval = self.poly.create_job(project, dedicated, self.ui.email.text())
        if retval:
            QMessageBox.critical(self, 'Error', retval)
            return
        return super().done(QDialog.Accepted)


class ThumbnailWidget(QWidget):
    """A custom widget for showing the status of a polygon x project.
    Displays the thumbnail if there is one, or explanatory text if not.
    """

    def __init__(self, project, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.project = project
        self._filename = None

        self.ui = Ui_Thumbnail()
        self.ui.setupUi(self)

    def load_pixmap(self, filename):
        if filename == self._filename:
            return
        pixmap = QPixmap(filename)
        self.ui.thumbnail.setPixmap(pixmap)
        self._filename = filename

    def update_poly(self, poly):
        thumb = poly.thumbnail(self.project)
        if thumb is not None:
            self.load_pixmap(thumb.filename)
            return
        self.ui.thumbnail.setPixmap(QPixmap())
        if poly.njobs(project=self.project) > 0:
            self.ui.thumbnail.setText(f'A job for {self.project} is currently running')
        else:
            self.ui.thumbnail.setText(f'No jobs running for {self.project}')


class KeyFilter(QObject):
    """A Qt key filter for intercepting keys that should be usable no
    matter the currently focused widget.
    """

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
    """The main GUI class."""

    def __init__(self):
        super().__init__()

        # The currently selected polygon
        self._poly = None

        # Project tab page widgets, one persistent object for each project
        self._project_tabs = {}

        # These attributes are used by the asynchronous job system
        self._thread = None
        self._worker = None
        self._continue = None

        # This class creates its own main window widget
        self.main = QMainWindow()
        self.setupUi(self.main)
        self.main.showMaximized()

        ConfigFile().verify(self)

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
        self.polylist.setModel(DatabaseModel(self.js_interface))
        self.polylist.selectionModel().selectionChanged.connect(self.polylist_selection_changed)
        self.polylist.doubleClicked.connect(self.polylist_double_clicked)

        # Create thumbnail widgets for each project
        for project, _ in PROJECTS:
            self._project_tabs[project] = ThumbnailWidget(project)
        self.projects.currentChanged.connect(self.project_tab_changed)

        # Main control buttons
        self.downloadbtn.clicked.connect(self.start_new_job)
        self.refreshbtn.clicked.connect(self.update_jobs)
        self.exportbtn.clicked.connect(self.export_data)

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
        """Update the polygon information widgets when the selected polygon is changed."""
        self._poly = poly
        if poly is None:
            self.polydetails.hide()
            self.downloadbtn.setEnabled(False)
            return

        self.polydetails.show()
        self.downloadbtn.setEnabled(True)

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
        """Hint that the currently selected polygon has new information about
        a given project. This may cause the tab page for the given
        project to be updated, shown or hidden. If select is true and
        the page is shown, it is also selected.
        """
        activate = self.poly.thumbnail(project) or self.poly.njobs(project=project) > 0
        widget = self._project_tabs[project]
        index = self.projects.indexOf(widget)

        # Make sure the widget itself is up to date
        if activate:
            widget.update_poly(self.poly)

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
        """Run some javascript code in the embedded webpage. If callback is
        given, it is called with the return value of the code.
        """
        if callback is None:
            self.webview.page().runJavaScript(code)
        else:
            self.webview.page().runJavaScript(code, callback)

    # When the project tab changes, update the enabled state of the export button
    def project_tab_changed(self, index):
        if self.poly is None or index == -1:
            self.exportbtn.setEnabled(False)
            return
        project = self.projects.widget(index).project
        has_data = self.poly.dedicated(project) or self.poly.ntiles(project) > 0
        self.exportbtn.setEnabled(bool(has_data))

    # When the web page has finished loading we can add the existing polygons
    # This will fail if done too soon
    def webview_finished_load(self):
        for poly in db:
            points = str(list(map(list, poly.geometry())))

            # The return value of the add_object javascript function is the internal
            # leaflet ID of the new object. This must be assigned to the poly.lfid
            # property.
            self.js(f'add_object({points})', partial(Polygon.lfid.fset, poly))

        # Pan and zoom the view so that all regions are visible
        self.webview.page().runJavaScript('focus_object(-1)')

    # When the user clicks on a region on the map, update the selected
    # item in the list.  This will trigger the selectionChanged signal,
    # which will update the detail view as well.
    def webview_selection_changed(self, lfid):
        if lfid < 0:
            index = QModelIndex()
        else:
            index = self.polylist.model().index(db.index(lfid=lfid), 0, QModelIndex())
        selection_model = self.polylist.selectionModel()
        selection_model.select(index, QItemSelectionModel.SelectCurrent)

    def webview_double_clicked(self, lfid):
        self.js(f'focus_object({lfid})')

    # When a new polygon was added, ask for its name and store it in the database.
    # This triggers a 'fake' selection changed signal as well.
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

    # A region has been selected: re-color the map view by running the
    # select_object javascript function, and update the 'poly'
    # attribute, thereby updating the detail view.
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

    # The signal handler for the 'new job' button.
    def start_new_job(self):
        if self.poly is None:
            return
        dialog = JobDialog(self.poly)
        retval = dialog.exec_()
        if retval == QDialog.Accepted:
            self.refresh_tabs_hint(dialog.project)

    # The signal handler for the 'export' button.
    def export_data(self):
        assert self.poly
        project = self.projects.currentWidget().project
        dialog = ExporterDialog(self.poly, project)
        retval = dialog.exec_()

    # Callbacks for the ConfigFile.verify function
    def message(self, title, text):
        QMessageBox.information(self.main, title, text)

    def query_str(self, title, text):
        name, accept = QInputDialog.getText(self.main, title, text)
        return name

    def run_thread(self, jobs, text, cont=None):
        """Run a number of jobs asynchronously.

        Each element of 'jobs' must be a function-like object which
        will be called in a separate thread. The return value must be
        another function-like object which is called in the original
        thread (see module geomaker.db.AsyncWorker).

        'Text' should be a short string explaining the work being
        done. If 'cont' is given, it is called after the thread has
        finished processing all the work.
        """

        # Only one asynchronous job at a time
        if self._thread:
            QMessageBox.critical(
                self.main, 'Job already running',
                'Geomaker is already processing in the background. ' +
                'Wait until it has finished before starting another.'
            )
            return

        # Shortcut in case there are no jobs
        if len(jobs) == 0:
            if cont is not None:
                cont()
            return

        # Create a thread and a worker object assigned to that thread
        thread = QThread()
        worker = Worker(jobs)
        worker.moveToThread(thread)

        # Connect signals so that we are notified when things happen
        thread.started.connect(worker.process)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._thread_finished)
        worker.callback.connect(self._thread_callback)

        # These objects will be garbage collected before they're ready
        # if we don't keep track of them
        self._thread = thread
        self._worker = worker
        self._continue = cont

        # Prepare the progress bar
        self.progress.setMaximum(len(jobs))
        self.progress.setValue(0)
        self.progress.setFormat(f'{text} (%v/{len(jobs)} · %p%)')

        thread.start()

    # Called whenever the currently working thread has produced a new result
    # This is a function-like object that should be called in the main thread
    def _thread_callback(self, result):
        # TODO: Find a good way to communicate back what has changed exactly
        poly, project = result()
        if poly is self.poly:
            self._project_tabs[project].update_poly(poly)
            self.refresh_tabs_hint(project, select=False)
        self.progress.setValue(self.progress.value() + 1)

    # Called whenever the currently working thread has finished all its jobs
    # This resets the progress bar and calls the continuation function, if given
    def _thread_finished(self):
        self._worker = None
        self._thread = None
        self.progress.setValue(0)
        self.progress.setMaximum(1)
        self.progress.setFormat('')
        if self._continue is not None:
            self._continue()

    def update_jobs(self):
        """Update all unfinished external jobs asynchronously.
        When finished, download data files for the jobs that are completed.
        """
        self.run_thread(
            [job.refresh(asynchronous=True) for job in db.jobs()],
            'Refreshing jobs', cont=self.download_jobs,
        )

    def download_jobs(self):
        """Download data files for completed jobs asynchronously."""
        self.run_thread(
            [job.download(asynchronous=True) for job in db.jobs(stage='complete')],
            'Downloading jobs',
        )


class Worker(QObject):
    """A lightweight asynchronous worker object designed to process a
    sequence of jobs.
    """

    # Emitted when processing has finished
    finished = pyqtSignal()

    # Emitted when a new result has been obtained
    callback = pyqtSignal(object)

    def __init__(self, workers):
        super().__init__()
        self.workers = workers

    @pyqtSlot()
    def process(self):
        for worker in self.workers:
            self.callback.emit(worker())
        self.finished.emit()


def main():
    """Primary GUI entry point."""

    app = QApplication(sys.argv)
    gui = GUI()
    sys.exit(app.exec_())
