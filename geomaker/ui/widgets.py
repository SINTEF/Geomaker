from PyQt5.QtCore import Qt, QItemSelectionModel
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QListView, QLabel


class QSelectedListView(QListView):
    """Hacked QListView that selects one and only one item at any time."""

    def setModel(self, model):
        super().setModel(model)
        self.setCurrentIndex(self.model().index(0))

    def selectionChanged(self, selected, deselected):
        if len(selected.indexes()) == 0 and len(deselected.indexes()) > 0:
            self.selectionModel().select(deselected.indexes()[0], QItemSelectionModel.Select)


class QResizeLabel(QLabel):
    """QLabel that automatically resizes its pixmap."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pixmap = QPixmap()

    def setPixmap(self, pixmap):
        self._pixmap = pixmap
        self._update_pixmap(True)

    def resizeEvent(self, *args, **kwargs):
        retval = super().resizeEvent(*args, **kwargs)
        self._update_pixmap(False)
        return retval

    def _update_pixmap(self, force):
        if self._pixmap.isNull() and force:
            super().setPixmap(self._pixmap)
        elif not self._pixmap.isNull():
            pixmap = self._pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            super().setPixmap(pixmap)
