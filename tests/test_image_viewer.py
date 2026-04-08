import numpy as np
import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage

from src.gui.image_viewer import ImageViewerDialog


class TestImageViewerDialog:
    def test_creates_and_shows(self, qtbot):
        """Dialog should open with an image without error."""
        img = np.full((100, 200, 3), 128, dtype=np.uint8)
        dialog = ImageViewerDialog(img)
        qtbot.addWidget(dialog)
        dialog.show()
        assert dialog.isVisible()
        dialog.close()

    def test_closes_on_escape(self, qtbot):
        """Pressing Escape should close the dialog."""
        img = np.full((100, 200, 3), 128, dtype=np.uint8)
        dialog = ImageViewerDialog(img)
        qtbot.addWidget(dialog)
        dialog.show()
        qtbot.keyClick(dialog, Qt.Key.Key_Escape)
        assert not dialog.isVisible()
