"""Settings → Speech model: which model is active (base or tuned to your voice), its
evaluation numbers, and switching between them.

Tuned models are folders in ``<config dir>/models/<name>`` written by
``scripts/finetune/install_model.py`` (ONNX files + ``eval.json``). The active model is the
``model`` config value: an onnx-asr name for the base model, or one of those folders.
Switching saves the config and restarts Murmr, so the model is loaded fresh.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QMessageBox, QPushButton, QVBoxLayout, QWidget

from . import config as config_mod
from .asr import BASE_MODEL, local_model_dir
from .widgets import setting_row

log = logging.getLogger(__name__)


@dataclass
class TunedModel:
    path: Path
    name: str
    report: dict = field(default_factory=dict)

    @property
    def recommended(self) -> bool:
        return bool(self.report.get("recommend"))

    def summary(self) -> str:
        """'Your voice 8.4% → 5.1% WER · general 6.0% → 6.1% · 42 min of your speech, Nov 12'."""
        b, t = self.report.get("base", {}), self.report.get("tuned", {})
        parts = []
        for key, label in (("yours", "Your voice"), ("general", "general speech")):
            if key in b and key in t:
                parts.append(f"{label} {b[key]['wer']:.1%} → {t[key]['wer']:.1%} WER")
        tr = self.report.get("training", {})
        if tr.get("user_minutes") is not None:
            when = f", {tr['created']}" if tr.get("created") else ""
            parts.append(f"trained on {tr['user_minutes']:.0f} min of your speech{when}")
        if not parts:
            return "Not evaluated yet (run scripts/finetune/evaluate.py)."
        return " · ".join(parts)


def models_dir() -> Path:
    return config_mod.config_dir() / "models"


def load_tuned(path: Path) -> TunedModel:
    report = {}
    try:
        report = json.loads((path / "eval.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        pass
    return TunedModel(path, path.name, report)


def installed_models(root: Path | None = None) -> list[TunedModel]:
    root = root or models_dir()
    if not root.is_dir():
        return []
    found = [load_tuned(p) for p in sorted(root.iterdir(), reverse=True) if local_model_dir(str(p))]
    return found


def same_path(a: str, b: Path) -> bool:
    try:
        return Path(a).resolve() == b.resolve()
    except (OSError, ValueError):
        return False


class SpeechModelSection(QWidget):
    """Rows for the Settings card. ``ctl`` needs cfg, update_setting() and restart();
    ``ctl.transcriber.fallback_reason`` (optional) says a tuned model failed to load."""

    def __init__(self, ctl, root: Path | None = None):
        super().__init__()
        self.ctl = ctl
        self.root = root
        self.lay = QVBoxLayout(self)
        self.lay.setContentsMargins(0, 0, 0, 0)
        self.lay.setSpacing(0)
        self.refresh()

    def refresh(self) -> None:
        from .window import clear_layout

        clear_layout(self.lay)
        current = self.ctl.cfg.model
        models = installed_models(self.root)
        active = next((m for m in models if same_path(current, m.path)), None)
        is_base = local_model_dir(current) is None and not Path(current).anchor

        if active is not None:
            title, desc = f"Tuned to your voice: {active.name}", active.summary()
        elif is_base:
            label = "Parakeet TDT 0.6B v3" if current == BASE_MODEL else current
            title, desc = f"{label} (base model)", "NVIDIA's original model, the same for everyone."
        else:
            title, desc = f"Custom: {current}", ""
        fallback = getattr(getattr(self.ctl, "transcriber", None), "fallback_reason", "")
        if fallback:
            desc = (f"Couldn't load {Path(current).name} ({fallback[:120]}), so the base model "
                    "is running. Switch to the base model to make that permanent.")
        button = None
        if not is_base:
            button = QPushButton("Use base model")
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.clicked.connect(lambda: self.switch(BASE_MODEL, "the base model"))
        row = setting_row("Active model", f"{title}\n{desc}" if desc else title, button, first=True)
        row.setObjectName("row")
        self.lay.addWidget(row)
        self.active_row = row

        for m in models:
            if m is active:
                continue
            use = QPushButton("Use")
            use.setCursor(Qt.CursorShape.PointingHandCursor)
            use.clicked.connect(lambda _=False, m=m: self.switch(str(m.path), m.name))
            note = "" if m.recommended else "  (evaluation didn't recommend it)"
            self.lay.addWidget(setting_row(f"Tuned model: {m.name}{note}", m.summary(), use))
        if not models:
            hint = QLabel("To tune the model to your voice: record on the Voice training page, then "
                          "follow scripts/finetune/README.md. Tuned models appear here.")
            hint.setObjectName("faint")
            hint.setWordWrap(True)
            hint.setContentsMargins(0, 0, 0, 12)
            self.lay.addWidget(hint)

    def switch(self, value: str, label: str, confirm: bool = True) -> None:
        if confirm and QMessageBox.question(
                self, "Switch speech model",
                f"Switch to {label}? Murmr restarts to load it (a few seconds).") \
                != QMessageBox.StandardButton.Yes:
            return
        err = self.ctl.update_setting("model", value)
        if err:
            QMessageBox.warning(self, "Switch speech model", err)
            return
        log.info("Speech model -> %s", value)
        self.ctl.restart()
