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
