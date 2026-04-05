from unittest.mock import MagicMock, patch


class TestNotifier:
    def test_no_crash_when_tray_unavailable(self):
        with patch("src.gui.notifier.QSystemTrayIcon") as MockTray:
            MockTray.isSystemTrayAvailable.return_value = False
            from src.gui.notifier import Notifier
            notifier = Notifier()
            notifier.notify("Title", "Body")

    def test_notify_calls_show_message_when_available(self):
        with patch("src.gui.notifier.QSystemTrayIcon") as MockTray:
            MockTray.isSystemTrayAvailable.return_value = True
            mock_icon = MagicMock()
            MockTray.return_value = mock_icon
            MockTray.MessageIcon = MagicMock()
            MockTray.MessageIcon.Information = 1
            from src.gui.notifier import Notifier
            notifier = Notifier()
            notifier.notify("处理完成", "/path/to/output.mp4")
            mock_icon.showMessage.assert_called_once_with(
                "处理完成", "/path/to/output.mp4", 1, 5000
            )

    def test_notify_error_uses_warning_icon(self):
        with patch("src.gui.notifier.QSystemTrayIcon") as MockTray:
            MockTray.isSystemTrayAvailable.return_value = True
            mock_icon = MagicMock()
            MockTray.return_value = mock_icon
            MockTray.MessageIcon = MagicMock()
            MockTray.MessageIcon.Warning = 2
            from src.gui.notifier import Notifier
            notifier = Notifier()
            notifier.notify_error("处理出错", "FFmpeg failed")
            mock_icon.showMessage.assert_called_once_with(
                "处理出错", "FFmpeg failed", 2, 5000
            )
