from whispa.mackeys import FN_FLAG, KEYCODES, key_names, modifier_change


def test_left_and_right_modifiers():
    # left ctrl down: device bit 0x1 + generic ctrl
    assert modifier_change(59, 0x40001, False) == ("press", {"ctrl_l", "ctrl"}, "ctrl_l")
    # right cmd down while left ctrl still held
    assert modifier_change(54, 0x140011, False) == ("press", {"cmd_r", "cmd"}, "cmd_r")
    # left ctrl released while right ctrl is held: generic bit still set, device bit gone
    assert modifier_change(59, 0x42000, False) == ("release", {"ctrl_l", "ctrl"}, "ctrl_l")


def test_modifier_without_device_bits_uses_generic_flag():
    assert modifier_change(58, 0x80000, False)[0] == "press"
    assert modifier_change(58, 0, False)[0] == "release"


def test_fn_key():
    assert modifier_change(63, FN_FLAG, False) == ("press", {"fn"}, "fn")
    assert modifier_change(63, 0, True) == ("release", {"fn"}, "fn")
    assert modifier_change(179, FN_FLAG, False)[0] == "press"  # globe key
    assert modifier_change(57, 0x10000, False) is None  # caps lock: ignored


def test_key_names_match_pynput_names():
    assert key_names(KEYCODES["space"]) == ({"space"}, "space")
    assert key_names(53) == ({"esc"}, "esc")
    assert key_names(9) == ({"v"}, "v")
    assert key_names(200) == ({"vk200"}, "vk200")


def test_stuck_modifier_is_released_on_the_next_key():
    from whispa.mackeys import FLAGS_CHANGED, KEY_DOWN, KeyTap

    events = []
    tap = KeyTap(lambda n, p: events.append(("press", p)), lambda n, p: events.append(("release", p)))
    tap._seen_key = True
    tap._handle(FLAGS_CHANGED, 59, 0x40001)  # left ctrl down
    assert events == [("press", "ctrl_l")]
    # its release was missed; the next key arrives with no ctrl flag at all
    tap._handle(KEY_DOWN, 15, 0x100)  # "r", no modifiers
    assert ("release", "ctrl_l") in events and events[-1] == ("press", "r")
    events.clear()
    tap._handle(KEY_DOWN, 15, 0x100)
    assert events == [("press", "r")]  # nothing left to heal


def test_held_modifiers_are_not_released_while_still_down():
    from whispa.mackeys import FLAGS_CHANGED, KEY_DOWN, KeyTap

    events = []
    tap = KeyTap(lambda n, p: events.append(("press", p)), lambda n, p: events.append(("release", p)))
    tap._seen_key = True
    tap._handle(FLAGS_CHANGED, 59, 0x40001)  # ctrl down
    tap._handle(FLAGS_CHANGED, 55, 0x140009)  # cmd down too
    tap._handle(KEY_DOWN, 15, 0x140009)  # ctrl+cmd+r
    assert not [e for e in events if e[0] == "release"]
