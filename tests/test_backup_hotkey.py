"""A second push-to-talk shortcut that works exactly like the first."""

from tests.test_responsive import window  # noqa: F401  (fixture)
from whispa.hotkey import Hotkey, HotkeyService, PTTStateMachine


class Clock:
    t = 100.0

    def __call__(self):
        return self.t


def make(dictate="ctrl+cmd", backup="alt_r", command="ctrl+alt+cmd"):
    events, clock = [], Clock()
    sm = PTTStateMachine(
        Hotkey.parse(dictate), command_hotkey=Hotkey.parse(command) if command else None,
        backup_hotkey=Hotkey.parse(backup) if backup else None,
        on_start=lambda: events.append(("start", sm.mode)), on_stop=lambda: events.append("stop"),
        on_cancel=lambda silent: events.append("cancel-silent" if silent else "cancel"),
        on_lock=lambda: events.append("lock"), clock=clock)
    return sm, events, clock


def test_backup_hold_and_release_dictates():
    sm, ev, c = make()
    sm.press({"alt_r", "alt"})
    c.t += 1
    sm.release({"alt_r", "alt"})
    assert ev == [("start", "dictate"), "stop"]


def test_main_shortcut_still_works():
    sm, ev, c = make()
    sm.press({"ctrl"}); sm.press({"cmd"})
    c.t += 1
    sm.release({"cmd"})
    assert ev == [("start", "dictate"), "stop"]


def test_backup_recording_isnt_ended_by_main_keys_being_absent():
    sm, ev, c = make()
    sm.press({"alt_r", "alt"})
    c.t += 1
    sm.press({"space"})  # hands-free lock while holding the backup
    assert ev == [("start", "dictate"), "lock"]
    sm.release({"alt_r", "alt", "space"})
    assert sm.state == sm.LOCKED


def test_either_shortcut_finishes_a_hands_free_recording():
    sm, ev, c = make()
    sm.press({"ctrl"}); sm.press({"cmd"}); sm.press({"space"})
    sm.release({"ctrl", "cmd", "space"})
    c.t += 2
    sm.press({"alt_r", "alt"})
    assert ev[-1] == "stop"


def test_double_tap_backup_locks():
    sm, ev, c = make()
    sm.press({"alt_r", "alt"}); c.t += 0.1; sm.release({"alt_r", "alt"})
    c.t += 0.1
    sm.press({"alt_r", "alt"})
    assert ev[-1] == "lock"


def test_command_mode_unaffected():
    sm, ev, c = make()
    sm.press({"ctrl"}); sm.press({"alt"}); sm.press({"cmd"})
    assert ev == [("start", "command")]


def test_backup_same_as_command_is_ignored():
    sm, ev, c = make(backup="ctrl+alt+cmd")
    sm.press({"ctrl"}); sm.press({"alt"}); sm.press({"cmd"})
    assert ev == [("start", "command")]


def test_set_live_and_turn_off():
    sm, ev, c = make(backup=None)
    svc = HotkeyService(sm)
    svc.set_hotkey(sm.hotkey, "space", backup_hotkey=Hotkey.parse("f8"))
    assert sm.command_hotkey is not None  # kept
    sm.press({"f8"}); c.t += 1; sm.release({"f8"})
    assert ev == [("start", "dictate"), "stop"]
    svc.set_hotkey(sm.hotkey, "space", backup_hotkey=None)
    sm.press({"f8"})
    assert ev == [("start", "dictate"), "stop"]


def test_settings_page_backup_row_and_off(window):
    _app, w = window
    page = w.settings
    calls = []
    page.ctl.update_setting = lambda key, value: calls.append((key, value))
    page.backup.set_labels(["F8"])
    assert page.backup._off is not None and not page.backup._off.isHidden()
    page._turn_off("backup_hotkey")
    assert ("backup_hotkey", "") in calls
    assert page.backup._is_off and page.backup._off.isHidden()
    assert page.hotkey._off is None  # the main shortcut can't be switched off
