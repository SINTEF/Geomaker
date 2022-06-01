import os
os.environ['QTWEBENGINE_DISABLE_SANDBOX'] = '1'

from contextlib import contextmanager
from functools import partial
from os.path import dirname, realpath, join
from operator import attrgetter
from pathlib import Path
import sys
import time

import numpy as np

from PyQt5.QtCore import (
    Qt, QObject, QEvent, QItemSelectionModel, QModelIndex, QThread,
    QUrl, pyqtSignal, pyqtSlot
)
from PyQt5.QtGui import QPixmap, QIcon, QKeyEvent, QImage, QFont
from PyQt5.QtWebChannel import QWebChannel
from PyQt5.QtWidgets import (
    QApplication, QDialog, QFileDialog, QInputDialog, QLayout,
    QMainWindow, QMessageBox, QProgressBar, QPushButton, QWidget,
)

from .asynchronous import ThreadManager, PipeJob, SequenceJob, AbstractJob
from .ui.utils import key_to_text, angle_to_degrees
from .ui.models import ProjectsModel, DatabaseModel
from .db import PROJECTS, Polygon, Database, Job
from .filesystem import ConfigFile, DataFile
from .util import to_latlon
from . import export

# Classes generated by qtdesigner
from .ui.interface import Ui_MainWindow
from .ui.thumbnail import Ui_Thumbnail
from .ui.jobdialog import Ui_JobDialog
from .ui.exporter import Ui_Exporter
from .ui.polygon import Ui_Polygon


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

    def emit(self, name, *args):
        getattr(self, name).emit(*args)

    # Every possible signal signature (with an added str in front)
    # must be listed here.  These functions are the only points of
    # interaction visible from Javascript for emitting events.
    @pyqtSlot(str, int, str)
    def emit_is(self, name, *args):
        self.emit(name, *args)

    @pyqtSlot(str, int)
    def emit_i(self, name, *args):
        self.emit(name, *args)


class ExporterDialog(QDialog):
    """A dialog window for exporting data."""

    FORMATS = [
        ('png', {'png'}, 'Portable Network Graphics (PNG)'),
        ('jpeg', {'jpg', 'jpeg'}, 'Joint Photographic Experts Group (JPEG)'),
        ('tiff', {'tif', 'tiff'}, 'Georeferenced Tagged Image File Format (GeoTIFF)'),
        ('g2', {'g2'}, 'GoTools B-Splines (G2)'),
        ('stl', {'stl'}, 'Stereolithography (STL)'),
        ('vtk', {'vtk'}, 'Visualization Toolkit Legacy (VTK)'),
        ('vtu', {'vtu'}, 'Visualization Toolkit Unstructured (VTU)'),
        ('vts', {'vts'}, 'Visualization Toolkit Structured (VTS)'),
    ]

    COORDS = [
        ('utm33n', 'UTM Zone 33 North', 'm'),
        ('latlon', 'Latitude and Longitude', '°'),
    ]

    def __init__(self, poly, project, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.poly = poly
        self.selected_filter = None
        self.ui = Ui_Exporter()
        self.ui.setupUi(self)

        self.threadmanager = ThreadManager(self)
        self._recompute_count = 0

        # Populate dropdown boxes
        self.ui.projects.addItems([project.name for project in poly.projects()])
        self.ui.projects.setCurrentIndex(next(i for i, p in enumerate(poly.projects()) if p is project))

        self.ui.formats.addItems([name for _, _, name in self.FORMATS])
        self.ui.coordinates.addItems([name for _, name, _ in self.COORDS])

        # The list of color maps requires some messing around
        boldfont = QFont(self.ui.colormaps.font())
        boldfont.setBold(True)
        for category, entries in export.iter_map_categories():
            if self.ui.colormaps.count() > 0:
                self.ui.colormaps.insertSeparator(self.ui.colormaps.count())

            self.ui.colormaps.addItem(category.upper())
            idx = self.ui.colormaps.count() - 1
            self.ui.colormaps.setItemData(idx, boldfont, Qt.FontRole)
            self.ui.colormaps.setItemData(idx, Qt.AlignCenter, Qt.TextAlignmentRole)
            item = self.ui.colormaps.model().item(idx)
            item.setFlags(item.flags() & ~Qt.ItemIsSelectable)

            self.ui.colormaps.addItems(entries)

        # Set the default settings based on the previous time
        data = DataFile()
        self.format = data.get('export-format', 'png')
        self.ui.filename.addItems(data.get('export-filenames', []))
        self.ui.resolution.setValue(data.get('export-resolution', 1.0))
        self.ui.zero_sea_level.setChecked(data.get('export-zero-sea', True))
        self.ui.textures.setChecked(data.get('export-textures', True))
        self.boundary_mode = data.get('export-boundary-mode', 'interior')
        self.rotation_mode = data.get('export-rotation-mode', 'north')
        self.axis_align = data.get('export-axis-align', False)
        self.coords = data.get('export-coords', 'utm33n')
        self.colormap = data.get('export-colormap', 'Terrain')
        self.ui.invertmap.setChecked(data.get('export-colormap-invert', False))
        self.ui.structured.setChecked(data.get('export-structured', False))
        self.offset = (data.get('export-offset-origin-x', 0), data.get('export-offset-origin-y', 0))

        if self.ui.filename.currentText().strip() == '':
            self.ui.filename.setEditText(poly.name)

        # Connect signals
        self.ui.projects.currentIndexChanged.connect(self.project_changed)
        self.ui.filename.currentTextChanged.connect(self.filename_changed)
        self.ui.browsebtn.clicked.connect(self.browse)
        self.ui.formats.currentIndexChanged.connect(self.format_changed)
        self.ui.coordinates.currentIndexChanged.connect(self.coords_changed)
        self.ui.colormaps.currentIndexChanged.connect(self.colormap_changed)
        self.ui.invertmap.stateChanged.connect(self.colormap_changed)
        self.ui.okbtn.clicked.connect(self.accept)
        self.ui.cancelbtn.clicked.connect(self.reject)
        self.ui.refreshbtn.clicked.connect(self.recompute)
        self.ui.offsetbtn.clicked.connect(self.recompute_offset)

        # Trigger some basic validation
        self.colormap_changed()
        self.format_changed()
        self.update_resolution_suffix()

        self.setFixedSize(self.size())

    @property
    def project(self):
        return self.poly.projects()[self.ui.projects.currentIndex()]

    @property
    def boundary_mode(self):
        if self.ui.true_geometry.isChecked():
            return 'actual'
        elif self.ui.interior_bnd.isChecked():
            return 'interior'
        return 'exterior'

    @boundary_mode.setter
    def boundary_mode(self, value):
        if value == 'interior':
            self.ui.interior_bnd.setChecked(True)
        elif value == 'exterior':
            self.ui.exterior_bnd.setChecked(True)
        else:
            self.ui.true_geometry.setChecked(True)

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
    def axis_align(self):
        return self.ui.axis_align.isChecked()

    @axis_align.setter
    def axis_align(self, value):
        self.ui.axis_align.setChecked(value)

    @property
    def offset(self):
        return self.ui.offsetx.value(), self.ui.offsety.value()

    @offset.setter
    def offset(self, value):
        self.ui.offsetx.setValue(value[0])
        self.ui.offsety.setValue(value[1])

    @property
    def coords(self):
        return self.COORDS[self.ui.coordinates.currentIndex()][0]

    @coords.setter
    def coords(self, value):
        try:
            self.ui.coordinates.setCurrentIndex(next(i for i, (v, _, _) in enumerate(self.COORDS) if v == value))
        except StopIteration:
            self.ui.coordinates.setCurrentIndex(0)

    @property
    def coords_unit(self):
        return self.COORDS[self.ui.coordinates.currentIndex()][2]

    @property
    def colormap(self):
        return self.ui.colormaps.currentText()

    @colormap.setter
    def colormap(self, value):
        try:
            self.ui.colormaps.setCurrentText(value)
        except:
            pass

    @property
    def format(self):
        return self.FORMATS[self.ui.formats.currentIndex()][0]

    @format.setter
    def format(self, value):
        try:
            self.ui.formats.setCurrentIndex(next(i for i, (fmt, _, _) in enumerate(self.FORMATS) if value == fmt))
        except StopIteration:
            self.ui.formats.setCurrentIndex(0)

    @property
    def format_suffixes(self):
        return self.FORMATS[self.ui.formats.currentIndex()][1]

    @property
    def rectangularize(self):
        return export.requires_rectangle(self.format)

    @property
    def image_mode(self):
        return export.is_image_format(self.format)

    @property
    def color_maps_enabled(self):
        return self.image_mode and self.project.ndims == 1

    def project_changed(self):
        self.ui.colormaps.setEnabled(self.color_maps_enabled)
        self.ui.invertmap.setEnabled(self.color_maps_enabled)

    def format_changed(self):
        self.ui.true_geometry.setEnabled(not self.rectangularize)
        self.ui.colormaps.setEnabled(self.color_maps_enabled)
        self.ui.invertmap.setEnabled(self.color_maps_enabled)
        self.ui.textures.setEnabled(export.supports_texture(self.format))
        self.ui.structured.setEnabled(export.supports_structured_choice(self.format))

        if self.rectangularize and self.ui.true_geometry.isChecked():
            self.ui.interior_bnd.setChecked(True)

        filename = Path(self.ui.filename.currentText())
        if filename.suffix[1:] not in self.format_suffixes:
            with block_signals(self.ui.filename) as obj:
                obj.setEditText(str(filename.with_suffix('.' + self.format)))

    def coords_changed(self):
        # Very basic conversion between degrees and meters:
        # originally, a kilometer was defined as one 10000th of the
        # distance between the equator and the pole, so a degree of
        # latitude is roughly 10000/90 meters.
        if self.coords_unit == '°' and self.ui.resolution.suffix() == 'm':
            self.ui.resolution.setValue(self.ui.resolution.value() / 10_000_000 * 90)
        elif self.coords_unit == 'm' and self.ui.resolution.suffix() == '°':
            self.ui.resolution.setValue(self.ui.resolution.value() / 90 * 10_000_000)
        self.update_resolution_suffix()

    def colormap_changed(self):
        data = export.preview_colormap(self.colormap, res=500, invert=self.ui.invertmap.isChecked())
        image = QImage(data.data, data.shape[1], data.shape[0], QImage.Format_RGBA8888)
        self.ui.colormap_preview.setPixmap(QPixmap(image).scaled(
            max(690, self.ui.colormap_preview.width()),
            self.ui.colormap_preview.height(),
        ))

    def update_resolution_suffix(self):
        self.ui.resolution.setSuffix(self.coords_unit)

    def recompute(self):
        if self.boundary_mode == 'actual':
            self.ui.fitwarning.setText('')
            return
        self.ui.fitwarning.setText('Computing...')
        self._recompute_count += 1
        job = PipeJob([
            Database().get_object(Polygon, self.poly.id),
            Polygon.check_area(mode=self.boundary_mode, rotate=self.rotation_mode, coords=self.coords)
        ])
        self.threadmanager.enqueue(
            job, priority='low',
            callback=partial(self._recompute_callback, self._recompute_count)
        )

    def _recompute_callback(self, count, result):
        if self._recompute_count != count:
            return
        pctg, theta = result
        if self.boundary_mode == 'interior':
            self.ui.fitwarning.setText(f'{100*pctg:.2f}% shortfall, rotated {theta*180/np.pi:.5f}°')
        else:
            self.ui.fitwarning.setText(f'{100*pctg:.2f}% overshoot, rotated {theta*180/np.pi:.5f}°')

    def recompute_offset(self):
        self.offset = self.poly.optimal_offset(self.coords)

    def browse(self):
        filters = [
            'Images (*.png *.jpg *.jpeg *.tif *.tiff)',
            'GoTools (*.g2)',
            'Stereolithography (*.stl)'
            'Visualization Toolkit (*.vtk *.vtu *.vts)'
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
                'export-textures': self.ui.textures.isChecked(),
                'export-structured': self.ui.structured.isChecked(),
                'export-colormap-invert': self.ui.invertmap.isChecked(),
                'export-colormap': self.colormap,
                'export-boundary-mode': self.boundary_mode,
                'export-rotation-mode': self.rotation_mode,
                'export-axis-align': self.axis_align,
                'export-offset-origin-x': self.offset[0],
                'export-offset-origin-y': self.offset[1],
            })

    def done(self, result):
        self.threadmanager.close()

        if result != QDialog.Accepted:
            return super().done(result)

        if not export.has_support(self.format):
            QMessageBox.critical(
                self, 'Error',
                f'Additional packages must be installed for {self.format.upper()} output.',
            )
            return

        self.update_datafile()

        export_job = export.export_job(
            project=self.project,
            boundary_mode=self.boundary_mode,
            rotation_mode=self.rotation_mode,
            axis_align=self.axis_align,
            coords=self.coords,
            resolution=self.ui.resolution.value(),
            format=self.format,
            structured=self.ui.structured.isChecked(),
            colormap=self.colormap,
            invert=self.ui.invertmap.isChecked(),
            texture=self.ui.textures.isChecked(),
            zero_sea_level=self.ui.zero_sea_level.isChecked(),
            offset_origin=self.offset,
            filename=self.ui.filename.currentText(),
        )

        self.job = PipeJob([
            Database().get_object(Polygon, self.poly.id),
            export_job,
        ])

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
        self.ui.projectlist.selectionModel().selectionChanged.connect(self.project_changed)
        self.ui.zoomslider.valueChanged.connect(self.zoom_changed)

        self.ui.okbtn.clicked.connect(self.accept)
        self.ui.cancelbtn.clicked.connect(self.reject)

        self.setFixedSize(self.size())
        self.project_changed()

        self.retval = None

    @property
    def project(self):
        return PROJECTS.values()[self.ui.projectlist.selectedIndexes()[0].row()]

    @property
    def dedicated(self):
        return self.ui.dedicated.isChecked()

    def project_changed(self):
        project = self.project
        self.ui.emaillabel.setVisible(project.supports_email)
        self.ui.email.setVisible(project.supports_email)
        self.ui.dedicated.setVisible(project.supports_dedicated)
        self.ui.zoomlabel.setVisible(bool(project.zoomlevels))
        self.ui.zoomslider.setVisible(bool(project.zoomlevels))

        if project.zoomlevels:
            lo, hi = project.zoomlevels
            self.ui.zoomslider.setMinimum(lo)
            self.ui.zoomslider.setMaximum(hi)

    def zoom_changed(self):
        value = self.ui.zoomslider.value()
        self.ui.zoomlabel.setText(f'Zoom level: {value}')

    def done(self, result):
        # If the dialog was canceled, no further validation
        if result != QDialog.Accepted:
            return super().done(result)

        if self.dedicated and self.poly.dedicated(self.project):
            answer = QMessageBox.question(
                self, 'Delete existing dedicated file?',
                'This region already has a dedicated data file. Delete it and download again?'
            )
            if answer == QMessageBox.No:
                return
            self.poly.delete_dedicated(self.project)

        if not self.dedicated and self.poly.ntiles(self.project) > 0:
            answer = QMessageBox.question(
                self, 'Disassociate existing tiles?',
                'This region already has associated tiles. Disassociate them and download again?'
            )
            if answer == QMessageBox.No:
                return
            self.poly.delete_tiles(self.project)

        if self.poly.job(self.project, self.dedicated):
            answer = QMessageBox.question(
                self, 'Delete existing job?',
                'This region already has a job of this type. Delete it and restart?'
            )
            if answer == QMessageBox.No:
                return
            self.poly.delete_job(self.project, self.dedicated)

        kwargs = {}
        if self.project.supports_dedicated:
            kwargs['dedicated'] = self.dedicated
        else:
            kwargs['dedicated'] = False
        if self.project.supports_email:
            kwargs['email'] = self.ui.email.text()
        if self.project.zoomlevels:
            kwargs['zoom'] = self.ui.zoomslider.value()

        retval = self.poly.create_job(self.project, **kwargs)
        if isinstance(retval, str):
            QMessageBox.critical(self, 'Error', retval)
            return

        self.retval = retval
        return super().done(QDialog.Accepted)


class PolygonDialog(QDialog):
    """A dialop for entering a new polygon."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.ui = Ui_Polygon()
        self.ui.setupUi(self)
        self.ui.coordinates.addItems([name for _, name, _ in ExporterDialog.COORDS])

        self.ui.buttonBox.accepted.connect(self.accept)
        self.ui.buttonBox.rejected.connect(self.accept)


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
        self._filename = None
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
        if text == 'DEL':
            self.ui.delete_current_polygon()
            return False
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

        # This class creates its own main window widget
        self.main = QMainWindow()
        self.setupUi(self.main)
        self.main.showMaximized()

        # Used for running jobs asynchronously
        self.threadmanager = ThreadManager(self.main)

        # Check that the config file has everything we need
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
        self.polylist.setModel(DatabaseModel(self))
        self.polylist.selectionModel().selectionChanged.connect(self.polylist_selection_changed)
        self.polylist.doubleClicked.connect(self.polylist_double_clicked)

        # Create thumbnail widgets for each project
        for project in PROJECTS.values():
            self._project_tabs[project] = ThumbnailWidget(project)
        self.projects.currentChanged.connect(self.project_tab_changed)

        # Main control buttons and menu
        self.downloadbtn.clicked.connect(self.start_new_job)
        self.refreshbtn.clicked.connect(self.update_jobs)
        self.exportbtn.clicked.connect(self.export_data)
        self.act_add_polygon.triggered.connect(self.add_polygon)

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
        for project in PROJECTS.values():
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
        activate = self.poly.is_project_active(project, pending=True)
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
        self.projects.insertTab(insertion_index, widget, project.key)
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
        for poly in Database():
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
            index = self.polylist.model().index(Database().index(lfid=lfid), 0, QModelIndex())
        selection_model = self.polylist.selectionModel()
        selection_model.select(index, QItemSelectionModel.SelectCurrent)

    def webview_double_clicked(self, lfid):
        self.js(f'focus_object({lfid})')

    # When a new polygon was added, ask for its name and store it in the database.
    # This triggers a 'fake' selection changed signal as well.
    def webview_polygon_added(self, lfid, data):
        name, accept = QInputDialog.getText(self.main, 'Name', 'Name this region:')
        if accept:
            Database().create(lfid, name, data)
            self.webview_selection_changed(lfid)
        else:
            self.js(f'remove_object({lfid})')

    def webview_polygon_edited(self, lfid, data):
        Database().update_points(lfid, data)
        self.webview_selection_changed(lfid)

    def webview_polygon_deleted(self, lfid):
        Database().delete(lfid)

    def delete_current_polygon(self):
        if self.poly:
            lfid = self.poly.lfid
            Database().delete(lfid)
            self.js(f'remove_object({lfid})')

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
        poly = Database()[index.row()]
        self.js(f'select_object({poly.lfid})')
        self.poly = poly

    def polylist_double_clicked(self, item):
        poly = Database()[item.row()]
        self.js(f'focus_object({poly.lfid})')

    # The signal handler for the 'new job' button.
    def start_new_job(self):
        if self.poly is None:
            return
        dialog = JobDialog(self.poly)
        retval = dialog.exec_()

        if retval == QDialog.Accepted:
            if isinstance(dialog.retval, AbstractJob):
                self.threadmanager.enqueue(dialog.retval, self.progress, priority='low')
            self.refresh_tabs_hint(dialog.project)

    # The signal handler for the 'export' button.
    def export_data(self):
        assert self.poly
        project = self.projects.currentWidget().project
        dialog = ExporterDialog(self.poly, project)
        retval = dialog.exec_()
        if retval == QDialog.Accepted:
            self.threadmanager.enqueue(dialog.job, self.progress, priority='low')

    # The signal handler for the 'add new polygon' menu item
    def add_polygon(self):
        dialog = PolygonDialog()
        retval = dialog.exec_()
        if retval != QDialog.Accepted:
            return

        name = dialog.ui.name.text()
        coords, *_ = ExporterDialog.COORDS[dialog.ui.coordinates.currentIndex()]
        try:
            points = [
                (float(a), float(b)) for a, b in
                (line.split(',') for line in dialog.ui.points.toPlainText().split('\n') if line.strip())
            ]
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Something went wrong with parsing the point list: {e}')
            return

        if coords != 'latlon':
            points = [to_latlon(pt, coords) for pt in points]

        ptstring = str(list(map(list, points)))
        self.js(f'add_object({ptstring})', partial(Database().create, name=name, data=points))

    # Callbacks for the ConfigFile.verify function
    def message(self, title, text):
        QMessageBox.information(self.main, title, text)

    def query_str(self, title, text):
        name, accept = QInputDialog.getText(self.main, title, text)
        return name

    def update_jobs(self):
        """Update all unfinished external jobs asynchronously.
        When finished, download data files for the jobs that are completed.
        """
        job = SequenceJob([job.refresh() for job in Database().jobs()])
        self.threadmanager.enqueue(job, self.progress, priority='low')

    def close(self):
        self.threadmanager.close()


def main():
    """Primary GUI entry point."""

    app = QApplication(sys.argv)
    gui = GUI()
    app.aboutToQuit.connect(gui.close)
    sys.exit(app.exec_())
