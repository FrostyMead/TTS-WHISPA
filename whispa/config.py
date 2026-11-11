"""User configuration: a JSON file in the platform config directory."""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

APP_NAME = "Murmr"
OLD_NAME = "Whispa"  # earlier name: existing installs keep their data folder
CONFIG_VERSION = 3
log = logging.getLogger(__name__)


def config_dir() -> Path:
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming")
        d, old = base / APP_NAME, base / OLD_NAME
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
        d, old = base / APP_NAME, base / OLD_NAME
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
        d, old = base / APP_NAME.lower(), base / OLD_NAME.lower()
    if not d.exists() and old.exists():
        return old  # installed before the rename: settings, history and dictionary stay put
    d.mkdir(parents=True, exist_ok=True)
    return d


def config_path() -> Path:
    return config_dir() / "config.json"


def default_command_hotkey() -> str:
    return "fn+ctrl" if sys.platform == "darwin" else "ctrl+alt+cmd"


def default_paste_last_hotkey() -> str:
    return "ctrl+cmd+v" if sys.platform == "darwin" else "alt+shift+z"


def default_read_hotkey() -> str:
    return "ctrl+cmd+r" if sys.platform == "darwin" else "alt+shift+r"


_EXCLUDED_APPS = ["1Password", "Bitwarden", "KeePass", "KeePassXC", "LastPass", "Dashlane",
                  "Keychain Access", "Enpass", "Proton Pass"]


def _default_styles() -> dict[str, str]:
    from .styles import DEFAULTS

    return dict(DEFAULTS)


def default_hotkey() -> str:
    # macOS exposes the fn/globe key to apps; Windows and Linux keyboards
    # handle fn in firmware, so the OS never sees it.
    return "fn" if sys.platform == "darwin" else "ctrl+cmd"


@dataclass
class LLMCleanup:
    """Local LLM polish (self-corrections, grammar). Default: Ollama on this machine."""

    enabled: str | bool = "auto"  # auto = use it when the server/model is available
    when: str = "smart"  # smart = only dictations that look like they need it | always
    provider: str = "ollama"  # ollama | openai (any OpenAI-compatible server)
    base_url: str = "http://127.0.0.1:11434"
    model: str = "gemma3:4b"
    api_key: str = ""
    keep_alive: str = "10m"  # how long Ollama keeps the model in GPU memory between uses
    timeout_s: float = 6.0
    prompt: str = ""  # empty = built-in prompt


@dataclass
class Config:
    # Keys
    hotkey: str = field(default_factory=default_hotkey)
    hands_free_key: str = "space"
    command_hotkey: str = field(default_factory=default_command_hotkey)  # "" disables Command Mode
    double_tap_to_lock: bool = True
    paste_last_hotkey: str = field(default_factory=default_paste_last_hotkey)  # "" = off
    read_hotkey: str = field(default_factory=default_read_hotkey)  # read the selection aloud
    # Speech recognition
    model: str = "nemo-parakeet-tdt-0.6b-v3"  # or a folder with an exported model (your tuned one)
    device: str = "auto"  # auto | cuda | directml | coreml | cpu
    quantization: str = "auto"  # auto (int8 on CPU, fp32 on GPU) | int8 | none
    # Audio
    input_device: str | int | None = None
    keep_mic_open: bool = False
    min_duration_ms: int = 300
    max_duration_s: int = 600
    # Text
    remove_fillers: bool = True
    smart_formatting: bool = True  # emails, websites, $, %, times written properly
    voice_commands: bool = True
    append_space: bool = True
    replacements: dict[str, str] = field(default_factory=dict)
    styles: dict[str, str] = field(default_factory=_default_styles)  # chat/email/code/other
    screen_context: bool = True  # spell names the way they appear in the window you dictate into
    llm_cleanup: LLMCleanup = field(default_factory=LLMCleanup)
    # Learning
    learn_from_edits: bool = True
    # Your voice (all of it stays on this computer). Opt-in: off until you switch it on.
    save_training_audio: bool = False  # "Improve recognition with my voice": keep dictation audio
    voice_clone: bool = False  # "Create a voice that sounds like me" from Voice training reads
    training_excluded_apps: list[str] = field(default_factory=lambda: list(_EXCLUDED_APPS))
    training_max_mb: int = 2000
    auto_tune: bool = False  # train overnight when enough new recordings have built up
    last_tune_minutes: float = 0.0  # recognition minutes at the last tuning run
    # Output
    paste_method: str = "auto"  # auto | clipboard | type
    no_text_field: str = "popup"  # popup = show copy card | paste = paste anyway
    paste_shortcut: str = ""  # empty = cmd+v on macOS, ctrl+v elsewhere
    restore_clipboard: bool = True
    # UI
    sounds: bool = True
    sound_volume: float = 0.12
    show_idle_pill: bool = True
    pill_anchor: str = "bottom-center"  # where the pill snaps; drag it or pick in Settings
    save_history: bool = True
    # Meetings / live captions
    meeting_mic: bool = True  # transcribe your microphone ("You")
    meeting_system: bool = True  # and the computer's audio ("Others"; Windows)
    # Read aloud (local Kokoro voice)
    read_voice: str = "af_heart"
    read_speed: float = 1.0
    update_check: bool = True  # look for a new version on GitHub every few hours (installed copies)
    auto_update: bool = True  # download it and restart when you're away (else: Settings > Updates)
    last_seen_version: str = ""  # for the "What's new" note after an update
    config_version: int = CONFIG_VERSION


def _from_dict(cls, data: dict[str, Any]):
    known = {f.name: f for f in fields(cls)}
    kwargs = {}
    for key, value in data.items():
        if key not in known:
            log.warning("Ignoring unknown config key %r", key)
            continue
        if key == "llm_cleanup" and isinstance(value, dict):
            value = _from_dict(LLMCleanup, value)
        kwargs[key] = value
    return cls(**kwargs)


def load(path: Path | None = None) -> Config:
    path = path or config_path()
    data: dict[str, Any] = {}
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            log.error("Could not read %s (%s); using defaults", path, exc)
            return Config()
    if data.get("config_version", 1) < 2:
        data.pop("llm_cleanup", None)  # v1 defaults pointed at a different API; reset
    if data.get("config_version", 1) < 3 and data.get("llm_cleanup", {}).get("keep_alive") == "30m":
        data["llm_cleanup"]["keep_alive"] = "10m"  # free GPU memory sooner for the speech model
    data["config_version"] = CONFIG_VERSION
    cfg = _from_dict(Config, data)
    for cat, style in _default_styles().items():  # categories added in later versions
        cfg.styles.setdefault(cat, style)
    # Write back so every option is visible and editable in the file.
    full = asdict(cfg)
    if full != data:
        save(cfg, path)
    return cfg


def save(cfg: Config, path: Path | None = None) -> None:
    path = path or config_path()
    path.write_text(json.dumps(asdict(cfg), indent=2) + "\n", encoding="utf-8")
