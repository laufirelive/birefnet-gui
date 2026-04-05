"""System notification wrapper using QSystemTrayIcon."""

from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QApplication, QSystemTrayIcon


class Notifier:
    """Sends system notifications. Degrades gracefully if tray unavailable."""

    def __init__(self):
        self._tray: QSystemTrayIcon | None = None
        if QSystemTrayIcon.isSystemTrayAvailable():
            self._tray = QSystemTrayIcon()
            app = QApplication.instance()
            if app and not app.windowIcon().isNull():
                self._tray.setIcon(app.windowIcon())
            else:
                self._tray.setIcon(QIcon())
            self._tray.setVisible(True)

    def notify(self, title: str, message: str) -> None:
        """Send an informational notification."""
        if self._tray is None:
            return
        self._tray.showMessage(
            title, message,
            QSystemTrayIcon.MessageIcon.Information, 5000,
        )

    def notify_error(self, title: str, message: str) -> None:
        """Send a warning notification."""
        if self._tray is None:
            return
        self._tray.showMessage(
            title, message,
            QSystemTrayIcon.MessageIcon.Warning, 5000,
        )
