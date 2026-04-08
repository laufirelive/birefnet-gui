import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QDialog,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QVBoxLayout,
)


class _ZoomableView(QGraphicsView):
    """QGraphicsView with scroll-to-zoom and drag-to-pan."""

    def __init__(self, scene: QGraphicsScene):
        super().__init__(scene)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setRenderHint(self.renderHints())
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

    def wheelEvent(self, event):
        factor = 1.15 if event.angleDelta().y() > 0 else 1.0 / 1.15
        self.scale(factor, factor)


class ImageViewerDialog(QDialog):
    """Full-screen-ish dialog for viewing an image with zoom and pan.

    Args:
        image: RGB uint8 numpy array (H, W, 3).
        parent: Optional parent widget.
    """

    def __init__(self, image: np.ndarray, parent=None):
        super().__init__(parent)
        self.setWindowFlags(
            Qt.WindowType.Dialog
            | Qt.WindowType.FramelessWindowHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._scene = QGraphicsScene(self)
        self._view = _ZoomableView(self._scene)
        self._view.setStyleSheet("background: #222222;")
        layout.addWidget(self._view)

        # Convert numpy RGB to QPixmap
        h, w, _ = image.shape
        qimage = QImage(image.data, w, h, w * 3, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)
        self._pixmap_item = QGraphicsPixmapItem(pixmap)
        self._scene.addItem(self._pixmap_item)

        # Size to 80% of screen, centered
        screen = self.screen().availableGeometry()
        dialog_w = int(screen.width() * 0.8)
        dialog_h = int(screen.height() * 0.8)
        self.resize(dialog_w, dialog_h)
        self.move(
            screen.x() + (screen.width() - dialog_w) // 2,
            screen.y() + (screen.height() - dialog_h) // 2,
        )

        # Fit image in view
        self._view.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)

    def showEvent(self, event):
        super().showEvent(event)
        self._view.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.close()
        else:
            super().keyPressEvent(event)

    def mouseDoubleClickEvent(self, event):
        self.close()
