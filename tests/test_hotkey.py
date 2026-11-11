import pytest

from whispa.hotkey import Hotkey, PTTStateMachine, normalize


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


@pytest.fixture
def sm():
    events = []
    clock = Clock()
    m = PTTStateMachine(
        Hotkey.parse("ctrl+win"),
        on_start=lambda: events.append("start"), on_stop=lambda: events.append("stop"),
        on_cancel=lambda silent: events.append("cancel-silent" if silent else "cancel"),
        on_lock=lambda: events.append("lock"), clock=clock)
    m.events, m.t = events, clock
    return m


def test_parse_aliases():
    assert Hotkey.parse("Ctrl + Win").keys == {"ctrl", "cmd"}
    assert Hotkey.parse("right_alt").keys == {"alt_r"}
    assert normalize("Option") == "alt"


def test_hold_to_talk(sm):
    sm.press({"ctrl", "ctrl_l"})
    assert sm.events == []
    sm.press({"cmd"})
    assert sm.events == ["start"]
    sm.press({"cmd"})  # auto-repeat
    sm.t.t += 2
    sm.release({"cmd"})
    assert sm.events == ["start", "stop"]
    sm.release({"ctrl", "ctrl_l"})
    assert sm.state == "idle"


def test_hands_free_lock_and_finish(sm):
    sm.press({"ctrl"}); sm.press({"cmd"})
    sm.t.t += 1
    sm.press({"space"}); sm.release({"space"})
    sm.release({"cmd"}); sm.release({"ctrl"})
    assert sm.events == ["start", "lock"]
    assert sm.state == "locked"
    sm.t.t += 5
    sm.press({"ctrl"}); sm.press({"cmd"})
    assert sm.events == ["start", "lock", "stop"]
    # releasing and pressing again is a fresh dictation
    sm.release({"cmd"}); sm.t.t += 1; sm.press({"cmd"})
    assert sm.events == ["start", "lock", "stop", "start"]


def test_no_restart_while_still_held_after_lock_finish(sm):
    sm.press({"ctrl"}); sm.press({"cmd"}); sm.press({"space"})
    sm.release({"space"}); sm.release({"cmd"}); sm.release({"ctrl"})
    sm.t.t += 3
    sm.press({"ctrl"}); sm.press({"cmd"})  # finish
    sm.t.t += 1
    sm.press({"space"})  # unrelated now that we're idle
    assert sm.events == ["start", "lock", "stop"]
    assert sm.state == "idle"


def test_double_tap_locks(sm):
    sm.press({"ctrl"}); sm.press({"cmd"})
    sm.t.t += 0.1
    sm.release({"cmd"})
    sm.t.t += 0.1
    sm.press({"cmd"})
    assert sm.events == ["start", "stop", "start", "lock"]
    assert sm.state == "locked"


def test_slow_second_press_does_not_lock(sm):
    sm.press({"ctrl"}); sm.press({"cmd"})
    sm.t.t += 1.0
    sm.release({"cmd"})
    sm.t.t += 0.1
    sm.press({"cmd"})
    assert sm.events == ["start", "stop", "start"]


def test_os_shortcut_cancels_silently(sm):
    sm.press({"ctrl"}); sm.press({"cmd"})
    sm.t.t += 0.1
    sm.press({"d"})  # Ctrl+Win+D = new virtual desktop
    assert sm.events == ["start", "cancel-silent"]
    sm.release({"d"}); sm.release({"cmd"}); sm.release({"ctrl"})
    assert sm.events == ["start", "cancel-silent"]


def test_typing_while_talking_is_ignored(sm):
    sm.press({"ctrl"}); sm.press({"cmd"})
    sm.t.t += 2
    sm.press({"d"})
    assert sm.events == ["start"]


def test_escape_cancels(sm):
    sm.press({"ctrl"}); sm.press({"cmd"})
    sm.t.t += 1
    sm.press({"esc"})
    assert sm.events == ["start", "cancel"]
    sm.release({"esc"}); sm.release({"cmd"})
    assert sm.events == ["start", "cancel"]


def test_click_toggle(sm):
    sm.toggle_hands_free()
    assert sm.events == ["start", "lock"]
    sm.toggle_hands_free()
    assert sm.events == ["start", "lock", "stop"]


def test_modifiers_held(sm):
    sm.press({"ctrl"})
    assert sm.modifiers_held()
    sm.release({"ctrl"})
    assert not sm.modifiers_held()
