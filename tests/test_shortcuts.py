import plistlib

from whispa.shortcuts import BUNDLE_ID, mac_app_files


def test_mac_app_bundle_contents():
    files = mac_app_files("/Users/me/.local/share/uv/python/cpython-3.13/bin/python3",
                          ["-I", "-S", "/Users/me/Library/Application Support/Whispa/runtime/boot.py"])
    info = plistlib.loads(files["Contents/Info.plist"])
    assert info["CFBundleIdentifier"] == BUNDLE_ID
    assert info["CFBundleExecutable"] == "Whispa"
    assert info["LSUIElement"] is True  # menu-bar only, no Dock icon
    assert "NSMicrophoneUsageDescription" in info
    script = files["Contents/MacOS/Whispa"].decode()
    assert script.startswith("#!/bin/bash")
    # the path with a space is quoted, and extra args are passed through
    assert "'/Users/me/Library/Application Support/Whispa/runtime/boot.py'" in script
    assert script.rstrip().endswith('"$@"')
