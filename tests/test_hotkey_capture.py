import pytest

from whispa.hotkey import Hotkey, HotkeyService, PTTStateMachine, combo_spec


def test_combo_modifiers_are_generic_and_ordered():
    assert combo_spec(["cmd", "ctrl_l"]) == "ctrl+cmd"
    assert combo_spec(["ctrl_r", "shift_l", "d"]) == "ctrl+shift+d"
    assert combo_spec(["alt_r", "space"]) == "alt+space"


def test_lone_modifier_is_side_specific():
    assert combo_spec(["alt_r"]) == "alt_r"
    assert combo_spec(["ctrl_r"]) == "ctrl_r"
    assert Hotkey.parse(combo_spec(["alt_r"])).label().startswith("Right")


def test_function_key_alone_ok():
    assert combo_spec(["f9"]) == "f9"


@pytest.mark.parametrize("keys", [["a"], ["space"], ["shift_l", "a"], ["a", "b"]])
def test_rejects_typing_keys(keys):
    with pytest.raises(ValueError):
        combo_spec(keys)


def test_single_mode():
    assert combo_spec(["space"], single=True) == "space"
    assert combo_spec(["ctrl_r"], single=True) == "ctrl"
    with pytest.raises(ValueError):
        combo_spec(["esc"], single=True)


def _service():
    events = []
    sm = PTTStateMachine(Hotkey.parse("ctrl+cmd"), on_start=lambda: events.append("start"),
                         on_stop=lambda: events.append("stop"), on_cancel=lambda s: None,
                         on_lock=lambda: None)
    return HotkeyService(sm), events


def test_capture_records_combo_without_starting_dictation():
    svc, events = _service()
    got = []
    svc.capture(lambda spec, err: got.append((spec, err)))
    svc._press({"ctrl", "ctrl_l"}, "ctrl_l")
    svc._press({"cmd", "cmd_l"}, "cmd")
    svc._release({"cmd", "cmd_l"}, "cmd")
    assert got == []  # still holding ctrl
    svc._release({"ctrl", "ctrl_l"}, "ctrl_l")
    assert got == [("ctrl+cmd", None)]
    assert events == []  # dictation never started
    # normal operation resumes
    svc._press({"ctrl"}, "ctrl_l")
    svc._press({"cmd"}, "cmd")
    assert events == ["start"]


def test_capture_escape_cancels():
    svc, _ = _service()
    got = []
    svc.capture(lambda spec, err: got.append((spec, err)))
    svc._press({"esc"}, "esc")
    assert got == [(None, None)]


def test_capture_reports_invalid():
    svc, _ = _service()
    got = []
    svc.capture(lambda spec, err: got.append((spec, err)))
    svc._press({"a"}, "a")
    svc._release({"a"}, "a")
    assert got[0][0] is None and "modifier" in got[0][1]


def test_set_hotkey_live():
    svc, events = _service()
    svc.set_hotkey(Hotkey.parse("alt_r"), "space")
    svc._press({"alt", "alt_r"}, "alt_r")
    assert events == ["start"]
