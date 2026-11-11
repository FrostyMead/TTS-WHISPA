import plistlib
from pathlib import Path

from whispa import shortcuts
from whispa.shortcuts import BUNDLE_ID, LAUNCHER_BIN, mac_app_files

PY = "/Users/me/.local/share/uv/python/cpython-3.13/bin/python3"
ARGS = ["-I", "-S", "/Users/me/Library/Application Support/Murmr/runtime/boot.py"]


def test_mac_app_uses_the_native_launcher():
    assert LAUNCHER_BIN.exists()
    head = LAUNCHER_BIN.read_bytes()[:4]
    assert head in (b"\xca\xfe\xba\xbe", b"\xbe\xba\xfe\xca")  # universal Mach-O
    files = mac_app_files(PY, ARGS)
    info = plistlib.loads(files["Contents/Info.plist"])
    assert info["CFBundleIdentifier"] == BUNDLE_ID
    assert info["CFBundleExecutable"] == "Murmr"
    assert info["LSUIElement"] is True  # menu-bar only, no Dock icon
    assert "NSMicrophoneUsageDescription" in info
    assert info["MurmrBundleFormat"] == shortcuts.BUNDLE_FORMAT
    assert files["Contents/MacOS/Murmr"] == LAUNCHER_BIN.read_bytes()
    # one argument per line, spaces kept as-is
    assert files["Contents/Resources/command"].decode().splitlines() == [PY, *ARGS]


def test_bundle_contents_do_not_change_with_the_app_version(monkeypatch):
    # macOS ties granted permissions to the bundle's exact contents.
    a = mac_app_files(PY, ARGS)
    monkeypatch.setattr("whispa.__version__", "9.9.9")
    assert mac_app_files(PY, ARGS) == a


def test_script_fallback_without_launcher(tmp_path):
    files = mac_app_files(PY, ARGS, launcher_bin=tmp_path / "missing")
    script = files["Contents/MacOS/Murmr"].decode()
    assert script.startswith("#!/bin/bash")
    assert "'/Users/me/Library/Application Support/Murmr/runtime/boot.py'" in script
    assert script.rstrip().endswith('"$@"')
    assert plistlib.loads(files["Contents/Info.plist"])["MurmrBundleFormat"] == 1


def _write_app(app: Path, files: dict) -> None:
    for rel, data in files.items():
        (app / rel).parent.mkdir(parents=True, exist_ok=True)
        (app / rel).write_bytes(data)


def test_outdated_detection(tmp_path, monkeypatch):
    monkeypatch.setattr(shortcuts, "launcher", lambda: (PY, ARGS))
    app = tmp_path / "Murmr.app"
    assert not shortcuts.mac_app_outdated(app)  # not installed: leave it alone
    _write_app(app, mac_app_files(PY, ARGS, launcher_bin=tmp_path / "missing"))
    assert shortcuts.mac_app_outdated(app)  # old shell-script bundle
    _write_app(app, mac_app_files(PY, ARGS))
    assert not shortcuts.mac_app_outdated(app)
    monkeypatch.setattr(shortcuts, "launcher", lambda: ("/other/python3", ARGS))
    assert shortcuts.mac_app_outdated(app)  # Python moved


def test_no_relaunch_inside_the_bundle(monkeypatch):
    monkeypatch.setenv("WHISPA_BUNDLE", "1")
    monkeypatch.setattr(shortcuts, "mac_app_usable", lambda: True)
    monkeypatch.setattr(shortcuts, "open_mac_app", lambda bg: (_ for _ in ()).throw(AssertionError))
    assert shortcuts.relaunch_through_mac_app(False) is False


def test_rename_moves_windows_shortcuts(tmp_path, monkeypatch):
    import sys

    import pytest

    if sys.platform != "win32":
        pytest.skip("Windows shortcuts")
    from whispa import autostart

    monkeypatch.setenv("APPDATA", str(tmp_path / "Roaming"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "home"))
    monkeypatch.setattr(autostart, "migrate_old", lambda before_app=False: False)  # no registry
    made = []
    monkeypatch.setattr(shortcuts, "install", lambda desktop=True: made.append(desktop))
    old = shortcuts._old_windows_shortcuts()
    for p in old:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("lnk")
    assert shortcuts.migrate_from_whispa() is False  # nothing to hand over on Windows
    assert made == [True] and not any(p.exists() for p in old)
    assert shortcuts.migrate_from_whispa() is False and made == [True]  # once only


def test_config_dir_keeps_the_old_folder(tmp_path, monkeypatch):
    import sys

    from whispa import config

    if sys.platform != "win32":
        import pytest
        pytest.skip("checked on Windows")
    monkeypatch.setenv("APPDATA", str(tmp_path))
    (tmp_path / "Whispa").mkdir()
    assert config.config_dir() == tmp_path / "Whispa"  # existing install: data stays put
    (tmp_path / "Whispa").rmdir()
    assert config.config_dir() == tmp_path / "Murmr"  # fresh install
