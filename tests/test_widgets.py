import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPoint, Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication, QWidget  # noqa: E402

from whispa.widgets import Switch, setting_row  # noqa: E402


def test_switch_stays_in_place_when_clicked():
    app = QApplication.instance() or QApplication([])
    host = QWidget()
    sw = Switch(False)
    row = setting_row("Title", "A description that is long enough to wrap " * 3, sw)
    row.setParent(host)
    host.resize(600, 120)
    row.resize(600, 120)
    host.show()
    app.processEvents()
    before = sw.geometry()
    assert before.x() > 400  # on the right
    QTest.mouseClick(sw, Qt.MouseButton.LeftButton, pos=QPoint(20, 11))
    end = time.monotonic() + 0.5  # let the knob animation run
    while time.monotonic() < end:
        app.processEvents()
    assert sw.isChecked()
    assert sw.geometry() == before  # the slide animation must not move the widget
