import sys
from PyQt6.QtWidgets import QApplication


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("BiRefNet Video Matting Tool")
    # MainWindow will be added in Task 6
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
