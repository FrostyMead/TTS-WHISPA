from whispa.hotkey import Hotkey, PTTStateMachine


class Clock:
    t = 100.0

    def __call__(self):
        return self.t


def make(dictate="ctrl+cmd", command="ctrl+alt+cmd"):
    events, clock = [], Clock()
    sm = PTTStateMachine(
        Hotkey.parse(dictate), command_hotkey=Hotkey.parse(command) if command else None,
        on_start=lambda: events.append(("start", sm.mode)), on_stop=lambda: events.append("stop"),
        on_cancel=lambda silent: events.append("cancel-silent" if silent else "cancel"),
        on_lock=lambda: events.append("lock"), clock=clock)
    return sm, events, clock


def test_command_hotkey_starts_command_mode():
    sm, ev, c = make()
    sm.press({"ctrl"}); sm.press({"alt"}); sm.press({"cmd"})
    assert ev == [("start", "command")]
    c.t += 2
    sm.release({"alt"})
    assert ev == [("start", "command"), "stop"]


def test_dictate_still_works_alongside():
    sm, ev, c = make()
    sm.press({"ctrl"}); sm.press({"cmd"})
    c.t += 1
    sm.release({"cmd"})
    assert ev == [("start", "dictate"), "stop"]


def test_late_extra_key_upgrades_to_command():
    sm, ev, c = make()
    sm.press({"ctrl"}); sm.press({"cmd"})  # dictate fires first...
    c.t += 0.1
    sm.press({"alt"})  # ...then Alt completes Ctrl+Alt+Win
    assert ev == [("start", "dictate"), "cancel-silent", ("start", "command")]
    assert sm.mode == "command"


def test_command_mode_has_no_hands_free_lock():
    sm, ev, c = make()
    sm.press({"ctrl"}); sm.press({"alt"}); sm.press({"cmd"})
    c.t += 1
    sm.press({"space"})
    assert "lock" not in ev and sm.state == "holding"


def test_disjoint_shortcuts():
    sm, ev, c = make(dictate="ctrl+`", command="ctrl+alt+cmd")
    sm.press({"ctrl"}); sm.press({"`"})
    c.t += 1
    sm.release({"`"}); sm.release({"ctrl"})
    sm.press({"ctrl"}); sm.press({"alt"}); sm.press({"cmd"})
    assert ev == [("start", "dictate"), "stop", ("start", "command")]


def test_no_command_hotkey():
    sm, ev, c = make(command=None)
    sm.press({"ctrl"}); sm.press({"alt"}); sm.press({"cmd"})
    assert ev == [("start", "dictate")]
