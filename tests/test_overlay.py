import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402
from PySide6.QtCore import QPoint, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QGuiApplication, QMouseEvent  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from whispa.overlay import ANCHORS, Overlay  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _mouse(widget, kind, global_pos: QPoint, button=Qt.MouseButton.LeftButton):
    local = QPointF(widget.mapFromGlobal(global_pos))
    buttons = Qt.MouseButton.NoButton if kind == QMouseEvent.Type.MouseButtonRelease else button
    ev = QMouseEvent(kind, local, QPointF(global_pos), button, buttons, Qt.KeyboardModifier.NoModifier)
    QApplication.sendEvent(widget, ev)


def _settle(app, ov, ms=400):
    import time

    end = time.monotonic() + ms / 1000
    while time.monotonic() < end:
        app.processEvents()
        ov._tick()
        time.sleep(0.01)


def _drag(app, ov, to: QPoint):
    start = ov.mapToGlobal(ov._pill_rect().center().toPoint())
    _mouse(ov, QMouseEvent.Type.MouseButtonPress, start)
    for i in range(1, 11):  # move in steps like a real drag
        p = start + (to - start) * i / 10
        _mouse(ov, QMouseEvent.Type.MouseMove, p)
    _mouse(ov, QMouseEvent.Type.MouseButtonRelease, to)
    _settle(app, ov)


@pytest.mark.parametrize("corner,expected", [
    ("top_right", "top-right"), ("top_left", "top-left"), ("left_mid", "middle-left"),
    ("bottom_right", "bottom-right"), ("top_mid", "top-center"), ("bottom_mid", "bottom-center"),
])
def test_drag_snaps_to_nearest_spot(app, corner, expected):
    ov = Overlay(lambda: 0.0, "hint")
    _settle(app, ov)
    area = QGuiApplication.primaryScreen().availableGeometry()
    targets = {
        "top_right": QPoint(area.right() - 60, area.top() + 50),
        "top_left": QPoint(area.left() + 70, area.top() + 40),
        "left_mid": QPoint(area.left() + 50, area.center().y() + 30),
        "bottom_right": QPoint(area.right() - 90, area.bottom() - 60),
        "top_mid": QPoint(area.center().x() + 20, area.top() + 60),
        "bottom_mid": QPoint(area.center().x() - 30, area.bottom() - 40),
    }
    got = []
    ov.anchor_changed.connect(got.append)
    _drag(app, ov, targets[corner])
    assert ov.anchor == expected and got == [expected]
    # It ended up exactly at the snap position, with the pill MARGIN from the edges.
    assert ov.pos() == ov._window_pos(expected, area)
    pill = ov._pill_rect().translated(QPointF(ov.pos()))
    v, h = expected.split("-")
    if h == "left":
        assert abs(pill.left() - (area.left() + ov.MARGIN)) <= 1
    if h == "right":
        assert abs((area.left() + area.width()) - pill.right() - ov.MARGIN) <= 1
    if v == "top":
        assert abs(pill.top() - (area.top() + ov.MARGIN)) <= 1
    ov.close()


def test_click_without_moving_is_still_a_click(app):
    ov = Overlay(lambda: 0.0, "hint")
    _settle(app, ov)
    clicks = []
    ov.idle_clicked.connect(lambda: clicks.append(1))
    p = ov.mapToGlobal(ov._pill_rect().center().toPoint())
    _mouse(ov, QMouseEvent.Type.MouseButtonPress, p)
    _mouse(ov, QMouseEvent.Type.MouseMove, p + QPoint(2, 1))  # tiny jitter, not a drag
    _mouse(ov, QMouseEvent.Type.MouseButtonRelease, p + QPoint(2, 1))
    assert clicks == [1] and ov.anchor == "bottom-center"
    ov.close()


def test_set_anchor_from_settings(app):
    ov = Overlay(lambda: 0.0, "hint", anchor="top-left")
    _settle(app, ov)
    area = QGuiApplication.primaryScreen().availableGeometry()
    assert ov.pos() == ov._window_pos("top-left", area)
    ov.set_anchor("middle-right")
    assert ov.pos() == ov._window_pos("middle-right", area)
    ov.set_anchor("nonsense")
    assert ov.anchor == "middle-right"
    assert set(ANCHORS) >= {"top-left", "middle-right"}
    ov.close()


@pytest.mark.parametrize("anchor,upright", [("middle-left", True), ("middle-right", True),
                                            ("bottom-center", False), ("top-right", False)])
def test_pill_stands_upright_on_side_edges(app, anchor, upright):
    ov = Overlay(lambda: 0.05, "hint", anchor=anchor)
    ov.set_state("listening")
    _settle(app, ov, 600)
    assert (ov._h > ov._w) == upright
    # hands-free buttons: top/bottom when upright, left/right otherwise
    ov.set_locked(True)
    _settle(app, ov, 600)
    rect = ov._pill_rect()
    cancel, stop = ov._cancel_rect(rect), ov._stop_rect(rect)
    if upright:
        assert cancel.center().y() < stop.center().y() and abs(cancel.center().x() - stop.center().x()) < 1
    else:
        assert cancel.center().x() < stop.center().x()
    # text still reads horizontally
    ov.flash("Learned: Sarah", seconds=5)
    _settle(app, ov, 600)
    assert ov._w > ov._h
    ov.close()


def test_drag_from_side_to_bottom_morphs_back(app):
    ov = Overlay(lambda: 0.05, "hint", anchor="middle-right")
    ov.set_state("listening")
    _settle(app, ov, 600)
    assert ov._h > ov._w
    area = QGuiApplication.primaryScreen().availableGeometry()
    _drag(app, ov, QPoint(area.center().x(), area.bottom() - 40))
    _settle(app, ov, 600)
    assert ov.anchor == "bottom-center" and ov._w > ov._h
    ov.close()
