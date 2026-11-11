"""Voice training: read sentences aloud to build a perfectly labeled dataset of your voice.

Each take is checked with Parakeet before it's saved: if what it heard is far from the
sentence (you misread it, or the mic was silent) you're asked to redo it, so bad labels
never reach the training set. Clips are saved through ``TrainingStore.add_read`` with
status ``read`` and the sentence itself as the label.
"""

from __future__ import annotations

import logging
import random
import re
import threading
import time
from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np
from PySide6.QtCore import QObject, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import (QApplication, QFrame, QHBoxLayout, QLabel, QPushButton,
                               QScrollArea, QSizePolicy, QVBoxLayout, QWidget)

from . import theme
from .config import Config
from .learn import Dictionary, TrainingStore

log = logging.getLogger(__name__)

TARGET_MINUTES = 60  # enough for a solid personal fine-tune; 30 already helps
MAX_TAKE_S = 30.0
MIN_TAKE_S = 0.8
SILENCE_RMS = 0.004  # loudest 50 ms frame below this: the mic heard nothing
MAX_WER = 0.35  # more than about one word in three wrong: probably misread


class VoiceTrainingController(Protocol):
    cfg: Config
    dictionary: Dictionary

    def training_store(self) -> TrainingStore: ...
    def transcribe_sample(self, audio: np.ndarray, sr: int) -> str: ...  # blocking, any thread


# --------------------------------------------------------------------------- checking a take
_NUMBER_WORDS = {w: str(i) for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
    "fifteen sixteen seventeen eighteen nineteen twenty".split())}


def normalize_words(text: str) -> list[str]:
    """Words for comparing a take with its sentence: case, punctuation and number style
    don't count ("3:30 PM" == "3.30 p.m.", "two" == "2", "25%" == "25 percent")."""
    t = text.lower().replace("’", "'")
    t = re.sub(r"\b([ap])\.\s?m\.", r"\1m", t)
    t = re.sub(r"(?<=\d)[,:.](?=\d)", "", t)
    t = t.replace("%", " percent ")
    t = re.sub(r"[^\w\s]", " ", t.replace("'", ""))
    return [_NUMBER_WORDS.get(w, w) for w in t.split()]


def word_error_rate(reference: str, hypothesis: str) -> float:
    ref, hyp = normalize_words(reference), normalize_words(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1] / len(ref)


def loudest_rms(audio: np.ndarray, sr: int) -> float:
    frame = max(1, int(sr * 0.05))
    n = len(audio) // frame
    if n == 0:
        return float(np.sqrt(np.mean(audio * audio))) if len(audio) else 0.0
    frames = audio[: n * frame].reshape(n, frame)
    return float(np.sqrt((frames * frames).mean(axis=1)).max())


@dataclass
class Verdict:
    ok: bool
    reason: str  # "" when ok; otherwise shown to the user
    wer: float = 0.0


def precheck(audio: np.ndarray, sr: int) -> Verdict | None:
    """Problems you can spot without running the model (too short, silent)."""
    seconds = len(audio) / sr if sr else 0.0
    if seconds < MIN_TAKE_S:
        return Verdict(False, "That was too short. Hold the button (or Space) for the whole "
                              "sentence.", 1.0)
    if loudest_rms(audio, sr) < SILENCE_RMS:
        return Verdict(False, "The microphone didn't pick anything up. Check the input device "
                              "in Settings → Microphone.", 1.0)
    return None


def check_take(prompt: str, heard: str, audio: np.ndarray, sr: int) -> Verdict:
    early = precheck(audio, sr)
    if early is not None:
        return early
    if not heard.strip():
        return Verdict(False, "Parakeet couldn't make out any words. Try again a little closer "
                              "to the mic.", 1.0)
    wer = word_error_rate(prompt, heard)
    if wer > MAX_WER:
        return Verdict(False, "What Parakeet heard is quite different from the sentence.", wer)
    return Verdict(True, "", wer)


# --------------------------------------------------------------------------- sentences
_TEMPLATES = (
    "Can you add {w} to the notes from today's meeting?",
    "I mentioned {w} in my last message, so let me know what you think.",
    "Let's check what {w} looks like before we make a decision.",
    "{w} came up again on the call this morning.",
    "I'll send you the details about {w} later this afternoon.",
    "Quick question about {w}: is that still on track for this week?",
    "Please double check the spelling of {w} in the final version.",
    "We talked about {w} for most of the afternoon.",
    "Could you remind me what we decided about {w}?",
    "I've updated the document so it says {w} everywhere.",
    "Thanks for the update on {w}, that's really helpful.",
    "Make sure {w} is on the agenda for Thursday.",
    "I'm still waiting to hear back about {w}.",
    "The next step is to talk to the team about {w}.",
    "Honestly, {w} is the part I'm most excited about.",
    "Do you have a minute to go over {w} with me?",
    "I wrote a short summary of {w} and shared it with everyone.",
    "Let me know if anything changes with {w}.",
)


def dictionary_words(dictionary: Dictionary) -> list[str]:
    """Your terms plus the targets of your corrections: the words you most need recognized."""
    terms, corrections = dictionary.snapshot()
    seen: dict[str, str] = {}
    for w in [*terms, *corrections.values()]:
        w = " ".join(w.split())
        if w and w.lower() not in seen and len(w) <= 40 and "\n" not in w:
            seen[w.lower()] = w
    return list(seen.values())


def dictionary_sentences(words: list[str], per_word: int = 3) -> list[str]:
    """A few different sentences around each word (stable for a given word)."""
    out = []
    for w in words:
        rng = random.Random(w.lower())
        for template in rng.sample(_TEMPLATES, min(per_word, len(_TEMPLATES))):
            s = template.format(w=w)
            out.append(s[0].upper() + s[1:])
    return out


def builtin_sentences() -> list[str]:
    try:
        from .reading_prompts import SENTENCES
    except ImportError:  # pragma: no cover
        return []
    return list(SENTENCES)


@dataclass
class Prompt:
    text: str
    source: str  # "builtin" | "dictionary"


class PromptQueue:
    """The sentences still to read: your dictionary words mixed in with the built-in set.

    The built-in order is shuffled once with a fixed seed, so every session continues where
    the last one stopped. When everything has been read, it starts again (a second take of
    a sentence is still useful audio).
    """

    DICT_EVERY = 3  # one dictionary sentence after every N built-in ones

    def __init__(self, builtin: list[str], from_dictionary: list[str], done: set[str]):
        order = list(dict.fromkeys(builtin))
        random.Random(1234).shuffle(order)
        self.round = 1
        base = [s for s in order if s not in done]
        mine = [s for s in dict.fromkeys(from_dictionary) if s not in done]
        if not base and not mine:  # everything read: go again
            self.round = 2
            base, mine = order, list(dict.fromkeys(from_dictionary))
        self.items: list[Prompt] = []
        for i, s in enumerate(base):
            if mine and i % self.DICT_EVERY == 0:
                self.items.append(Prompt(mine.pop(0), "dictionary"))
            self.items.append(Prompt(s, "builtin"))
        self.items += [Prompt(s, "dictionary") for s in mine]

    @property
    def current(self) -> Prompt | None:
        return self.items[0] if self.items else None

    def advance(self) -> None:
        if self.items:
            self.items.pop(0)

    def skip(self) -> None:
        """Move the current sentence to the end of the queue."""
        if len(self.items) > 1:
            self.items.append(self.items.pop(0))

    def push_front(self, prompt: Prompt) -> None:
        self.items.insert(0, prompt)

    def __len__(self) -> int:
        return len(self.items)


# --------------------------------------------------------------------------- widgets
class Bar(QWidget):
    """A thin rounded progress bar (also used as the live input-level meter)."""

    def __init__(self, height: int = 8):
        super().__init__()
        self.value = 0.0
        self.setFixedHeight(height)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.color: str | None = None

    def set_value(self, v: float) -> None:
        v = max(0.0, min(1.0, v))
        if abs(v - self.value) > 1e-3:
            self.value = v
            self.update()

    def paintEvent(self, _event) -> None:
        pal = theme.current()
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)
        r = self.height() / 2
        p.setBrush(QColor(pal.border))
        p.drawRoundedRect(QRectF(0, 0, self.width(), self.height()), r, r)
        if self.value > 0:
            p.setBrush(QColor(self.color or pal.accent))
            p.drawRoundedRect(QRectF(0, 0, max(self.height(), self.width() * self.value),
                                     self.height()), r, r)


class _Signals(QObject):
    checked = Signal(object)  # Take


@dataclass
class Take:
    prompt: Prompt
    audio: np.ndarray
    sr: int
    heard: str = ""
    verdict: Verdict | None = None
    error: str = ""


def _label(text: str, name: str = "", wrap: bool = True) -> QLabel:
    lab = QLabel(text)
    if name:
        lab.setObjectName(name)
    lab.setWordWrap(wrap)
    return lab


class VoiceTrainingPage(QScrollArea):
    """Read a sentence, hold to record, release: it's checked and saved."""

    def __init__(self, ctl: VoiceTrainingController,
                 recorder_factory: Callable[[], object] | None = None):
        from .window import _page, _section

        super().__init__()
        self.ctl = ctl
        self._recorder_factory = recorder_factory or self._default_recorder
        self._recorder = None
        self._queue: PromptQueue | None = None
        self._state = "idle"  # idle | recording | checking | flagged
        self._take: Take | None = None
        self._last_saved: tuple[str, Prompt, str] | None = None  # (sample id, prompt, heard)
        self._session = [0, 0.0]  # sentences, seconds
        self._t0 = 0.0
        self._filter_on = False
        self._signals = _Signals()
        self._signals.checked.connect(self._on_checked)

        area, lay = _page("Voice training",
                          "Read sentences aloud to teach the speech model your voice and accent. "
                          "Each one is saved with its exact text, so it's a perfect training "
                          "example. Everything stays on this computer.")
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setWidget(area.takeWidget())

        # Progress
        top = QHBoxLayout()
        self.minutes = _label("", "statValue")
        top.addWidget(self.minutes, 1)
        self.session_info = _label("", "faint")
        top.addWidget(self.session_info, 0, Qt.AlignmentFlag.AlignBottom)
        lay.addLayout(top)
        self.progress = Bar(8)
        lay.addWidget(self.progress)
        self.progress_note = _label("", "faint")
        lay.addWidget(self.progress_note)

        # Tune / train with what you've recorded (Windows + WSL); hidden for test controllers
        self.tuning = None
        if hasattr(ctl, "recorded_minutes"):
            from .tuning_ui import TuneSection

            self.tuning = TuneSection(ctl)
            lay.addWidget(self.tuning)

        # The sentence
        sec = _section(lay, "Read this sentence")
        sec.setContentsMargins(18, 16, 18, 16)
        sec.setSpacing(10)
        self.tag = _label("", "faint", wrap=False)
        sec.addWidget(self.tag)
        self.sentence = _label("")
        self.sentence.setObjectName("sentence")
        self.sentence.setStyleSheet("font-size: 21px; font-weight: 600;")
        self.sentence.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)
        self.sentence.setMinimumHeight(64)
        sec.addWidget(self.sentence)

        controls = QHBoxLayout()
        controls.setSpacing(12)
        self.record = QPushButton("Hold to record")
        self.record.setObjectName("primary")
        self.record.setMinimumHeight(40)
        self.record.setMinimumWidth(170)
        self.record.setCursor(Qt.CursorShape.PointingHandCursor)
        self.record.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.record.pressed.connect(self.start_take)
        self.record.released.connect(self.stop_take)
        controls.addWidget(self.record)
        meter = QVBoxLayout()
        meter.setSpacing(4)
        self.level = Bar(6)
        self.level.setVisible(False)
        self.hint = _label("or hold Space. Read at your normal pace and don't say the "
                           "punctuation.", "muted")
        meter.addWidget(self.hint)
        meter.addWidget(self.level)
        controls.addLayout(meter, 1)
        self.skip = QPushButton("Skip")
        self.skip.setToolTip("Skip this sentence for now (it comes back later)")
        self.skip.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.skip.clicked.connect(self.skip_prompt)
        controls.addWidget(self.skip)
        sec.addLayout(controls)

        # Result of the last take
        self.result = QFrame()
        self.result.setObjectName("row")
        self.result.setProperty("first", "false")
        rl = QVBoxLayout(self.result)
        rl.setContentsMargins(0, 12, 0, 0)
        rl.setSpacing(6)
        self.result_title = _label("")
        self.result_title.setStyleSheet("font-weight: 600;")
        self.result_heard = _label("", "muted")
        rl.addWidget(self.result_title)
        rl.addWidget(self.result_heard)
        actions = QHBoxLayout()
        actions.setSpacing(8)
        self.retry = QPushButton("Try again")
        self.retry.setObjectName("primary")
        self.retry.clicked.connect(self._retry)
        self.save_anyway = QPushButton("Save anyway")
        self.save_anyway.setToolTip("You read it right and Parakeet misheard you: these are the "
                                    "most useful clips")
        self.save_anyway.clicked.connect(self._save_anyway)
        self.redo = QPushButton("Re-record that one")
        self.redo.clicked.connect(self.redo_last)
        for b in (self.retry, self.save_anyway, self.redo):
            b.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            actions.addWidget(b)
        actions.addStretch(1)
        rl.addLayout(actions)
        sec.addWidget(self.result)
        self.result.setVisible(False)

        tips = _section(lay, "Tips")
        tips.setContentsMargins(18, 12, 18, 12)
        for t in ("Use the microphone and room you normally dictate in, at your usual distance.",
                  "Short sessions are fine: progress is saved after every sentence.",
                  "About 30 minutes already helps; 60 minutes or more gives the best results. "
                  "Then run the fine-tuning steps in scripts/finetune/README.md.",
                  "Sentences built from your Dictionary words are mixed in, because those are "
                  "the words you most need recognized."):
            tips.addWidget(_label("•  " + t, "muted"))
        lay.addStretch(1)

        self._meter = QTimer(self)
        self._meter.setInterval(40)
        self._meter.timeout.connect(self._tick)

    # -- recorder ---------------------------------------------------------
    def _default_recorder(self):
        from .audio import Recorder

        return Recorder(self.ctl.cfg.input_device, high_quality=True)  # 48 kHz: a voice needs it

    # -- keyboard: hold Space to record ------------------------------------
    def showEvent(self, e) -> None:
        super().showEvent(e)
        if not self._filter_on and QApplication.instance():
            QApplication.instance().installEventFilter(self)
            self._filter_on = True

    def hideEvent(self, e) -> None:
        super().hideEvent(e)
        if self._filter_on and QApplication.instance():
            QApplication.instance().removeEventFilter(self)
            self._filter_on = False
        if self._state == "recording":
            self.cancel_take()

    def eventFilter(self, obj, event) -> bool:
        et = event.type()
        if et in (event.Type.KeyPress, event.Type.KeyRelease) and self.isVisible() \
                and self.window().isActiveWindow():
            key = event.key()
            if key == Qt.Key.Key_Space:
                if not event.isAutoRepeat():
                    (self.start_take if et == event.Type.KeyPress else self.stop_take)()
                return True
            if key == Qt.Key.Key_Escape and et == event.Type.KeyPress and self._state == "recording":
                self.cancel_take()
                return True
        return False

    # -- data -------------------------------------------------------------
    def store(self) -> TrainingStore:
        return self.ctl.training_store()

    def refresh(self) -> None:
        if self.tuning is not None:
            self.tuning.refresh()
        if self._queue is None:
            done = self.store().read_prompts()
            words = dictionary_words(self.ctl.dictionary)
            self._queue = PromptQueue(builtin_sentences(), dictionary_sentences(words), done)
        self._update_progress()
        if self._state in ("idle", "flagged"):
            self._show_prompt()

    def _update_progress(self) -> None:
        seconds = self.store().read_seconds()
        minutes = seconds / 60
        self.minutes.setText(f"{minutes:.1f} of {TARGET_MINUTES} min recorded")
        self.progress.set_value(minutes / TARGET_MINUTES)
        n, s = self._session
        self.session_info.setText(f"This session: {n} sentence{'s' if n != 1 else ''} · "
                                  f"{s / 60:.1f} min" if n else "")
        left = len(self._queue) if self._queue else 0
        if minutes >= TARGET_MINUTES:
            note = "Target reached. More is still welcome, or go ahead and fine-tune."
        elif minutes >= 30:
            note = "Enough for a first fine-tune. More minutes make it better."
        else:
            note = f"About {max(1, round((30 - minutes)))} more minutes for a first fine-tune."
        if self._queue and self._queue.round > 1:
            note += " You've read every sentence once; this is a second round."
        self.progress_note.setText(f"{note}  {left} sentences in the queue.")

    def _consented(self) -> bool:
        cfg = self.ctl.cfg
        return bool(cfg.save_training_audio or getattr(cfg, "voice_clone", False))

    def _show_prompt(self) -> None:
        if not self._consented():
            self.sentence.setText("First choose what your recordings are for: Settings → Your "
                                  "voice → “Improve recognition with my voice” and/or “Create a "
                                  "voice that sounds like me”. They stay on this computer.")
            self.tag.setVisible(False)
            self.record.setEnabled(False)
            self.skip.setEnabled(False)
            return
        p = self._queue.current if self._queue else None
        if p is None:
            self.sentence.setText("No sentences left. Add words to your Dictionary for more.")
            self.tag.setText("")
            self.record.setEnabled(False)
            self.skip.setEnabled(False)
            return
        self.record.setEnabled(True)
        self.skip.setEnabled(len(self._queue) > 1)
        self.sentence.setText(p.text)
        self.tag.setText("From your Dictionary" if p.source == "dictionary" else "")
        self.tag.setVisible(p.source == "dictionary")

    # -- recording --------------------------------------------------------
    def start_take(self) -> None:
        if (self._state in ("recording", "checking") or not self._queue or not self._queue.current
                or not self._consented()):
            return
        try:
            if self._recorder is None:
                self._recorder = self._recorder_factory()
            self._recorder.start()
        except Exception as exc:  # noqa: BLE001
            log.error("Voice training: mic start failed: %s", exc)
            self._show_result("error", "Couldn't open the microphone.", str(exc))
            return
        self._state = "recording"
        self._t0 = time.monotonic()
        self.record.setText("Recording… release when done")
        self.record.setObjectName("recording")
        self._repolish(self.record)
        self.level.setVisible(True)
        self.hint.setText("Listening")
        self.result.setVisible(False)
        self.skip.setEnabled(False)
        self._meter.start()

    def _tick(self) -> None:
        if self._state != "recording" or self._recorder is None:
            return
        self.level.set_value(min(1.0, (self._recorder.level or 0.0) * 12))
        elapsed = time.monotonic() - self._t0
        self.hint.setText(f"Listening · {elapsed:.1f}s")
        if elapsed >= MAX_TAKE_S:
            self.stop_take()

    def _end_recording(self):
        self._meter.stop()
        self.level.setVisible(False)
        self.level.set_value(0)
        self.record.setText("Hold to record")
        self.record.setObjectName("primary")
        self._repolish(self.record)
        return self._recorder.stop()

    def cancel_take(self) -> None:
        if self._state != "recording":
            return
        self._end_recording()
        self._state = "idle"
        self.hint.setText("Cancelled.")
        self._show_prompt()

    def stop_take(self) -> None:
        if self._state != "recording":
            return
        audio, sr = self._end_recording()
        take = Take(self._queue.current, np.asarray(audio, np.float32), sr)
        self._state = "checking"
        self.record.setEnabled(False)
        self.hint.setText("Checking…")
        threading.Thread(target=self._check, args=(take,), name="voice-check", daemon=True).start()

    def _check(self, take: Take) -> None:
        """Worker thread: run Parakeet on the take and compare it with the sentence."""
        try:
            take.verdict = precheck(take.audio, take.sr)
            if take.verdict is None:
                take.heard = self.ctl.transcribe_sample(take.audio, take.sr)
                take.verdict = check_take(take.prompt.text, take.heard, take.audio, take.sr)
        except Exception as exc:  # noqa: BLE001
            log.exception("Voice training check failed")
            take.error = str(exc) or exc.__class__.__name__
        self._signals.checked.emit(take)

    def _on_checked(self, take: Take) -> None:
        self._state = "idle"
        self.record.setEnabled(True)
        self.skip.setEnabled(len(self._queue) > 1)
        self.hint.setText("or hold Space. Read at your normal pace and don't say the punctuation.")
        if take.error:
            self._take = take
            self._state = "flagged"
            self._show_result("error", "Couldn't check that take.", take.error)
            return
        if take.verdict.ok:
            self._save(take)
            return
        self._take = take
        self._state = "flagged"
        heard = f"Heard: “{take.heard}”" if take.heard else ""
        self._show_result("flagged", take.verdict.reason, heard,
                          can_save=bool(take.heard) and len(take.audio) / take.sr >= MIN_TAKE_S)

    def _save(self, take: Take) -> None:
        sid = self.store().add_read(take.audio, take.sr, take.prompt.text, take.heard)
        if sid is None:
            self._show_result("error", "Couldn't save the recording.", "See whispa.log.")
            return
        seconds = len(take.audio) / take.sr
        self._session[0] += 1
        self._session[1] += seconds
        self._last_saved = (sid, take.prompt, take.heard)
        self._take = None
        self._queue.advance()
        log.info("Voice training: saved %s (%.1fs)", sid, seconds)
        self._show_result("saved", f"Saved the previous sentence ({seconds:.1f}s).",
                          f"Heard: “{take.heard}”" if take.heard else "")
        self._update_progress()
        self._show_prompt()

    def _show_result(self, kind: str, title: str, detail: str, can_save: bool = False) -> None:
        pal = theme.current()
        color = {"saved": pal.success, "flagged": "#d97706", "error": pal.danger}[kind]
        mark = {"saved": "✓", "flagged": "!", "error": "✕"}[kind]
        self.result_title.setText(f"{mark}  {title}")
        self.result_title.setStyleSheet(f"font-weight: 600; color: {color};")
        self.result_heard.setText(detail)
        self.result_heard.setVisible(bool(detail))
        self.retry.setVisible(kind != "saved")
        self.save_anyway.setVisible(kind == "flagged" and can_save)
        self.redo.setVisible(kind == "saved")
        self.result.setVisible(True)

    def _retry(self) -> None:
        self._take = None
        self._state = "idle"
        self.result.setVisible(False)
        self._show_prompt()

    def _save_anyway(self) -> None:
        if self._take is not None:
            take, self._state = self._take, "idle"
            self._save(take)

    def redo_last(self) -> None:
        """Delete the last saved clip and read that sentence again."""
        if not self._last_saved or self._state == "recording":
            return
        sid, prompt, _ = self._last_saved
        self.store().remove(sid)
        self._last_saved = None
        self._session[0] = max(0, self._session[0] - 1)
        self._queue.push_front(prompt)
        self._state = "idle"
        self.result.setVisible(False)
        self._update_progress()
        self._show_prompt()

    def skip_prompt(self) -> None:
        if self._state in ("recording", "checking") or not self._queue:
            return
        self._take = None
        self._state = "idle"
        self._queue.skip()
        self.result.setVisible(False)
        self._update_progress()
        self._show_prompt()

    @staticmethod
    def _repolish(w: QWidget) -> None:
        w.style().unpolish(w)
        w.style().polish(w)
