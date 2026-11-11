"""User configuration: a JSON file in the platform config directory."""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

APP_NAME = "Whispa"
CONFIG_VERSION = 2
log = logging.getLogger(__name__)


def config_dir() -> Path:
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming")
        d = base / APP_NAME
    elif sys.platform == "darwin":
        d = Path.home() / "Library" / "Application Support" / APP_NAME
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
        d = base / APP_NAME.lower()
    d.mkdir(parents=True, exist_ok=True)
    return d


def config_path() -> Path:
    return config_dir() / "config.json"


def default_command_hotkey() -> str:
    return "fn+ctrl" if sys.platform == "darwin" else "ctrl+alt+cmd"


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
    keep_alive: str = "30m"  # how long Ollama keeps the model in memory between uses
    timeout_s: float = 6.0
    prompt: str = ""  # empty = built-in prompt


@dataclass
class Config:
    # Keys
    hotkey: str = field(default_factory=default_hotkey)
    hands_free_key: str = "space"
    command_hotkey: str = field(default_factory=default_command_hotkey)  # "" disables Command Mode
    double_tap_to_lock: bool = True
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
    voice_commands: bool = True
    append_space: bool = True
    replacements: dict[str, str] = field(default_factory=dict)
    llm_cleanup: LLMCleanup = field(default_factory=LLMCleanup)
    # Learning
    learn_from_edits: bool = True
    save_training_audio: bool = True
    training_max_mb: int = 2000
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
        data["config_version"] = CONFIG_VERSION
    cfg = _from_dict(Config, data)
    # Write back so every option is visible and editable in the file.
    full = asdict(cfg)
    if full != data:
        save(cfg, path)
    return cfg


def save(cfg: Config, path: Path | None = None) -> None:
    path = path or config_path()
    path.write_text(json.dumps(asdict(cfg), indent=2) + "\n", encoding="utf-8")
