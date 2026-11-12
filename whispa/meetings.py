"""Meeting notes and live captions: transcribe your mic and the computer's audio as it happens.

* Each source is its own channel: your microphone is "You", the computer's audio (the
  people in your Teams / Zoom / Meet call) is "Others". Within the computer's audio (or the
  mic, when it's the only source and picks up a whole room) voices are told apart:
  Person 1, Person 2, ... (speakers.py), which you can rename afterwards.
* Every ~1.5 s the not-yet-final audio of each channel is transcribed and shown as a live
  line. At a pause the line becomes final and its audio is dropped, so memory stays small
  however long the meeting is. No audio is saved, only the text.
* The pace adapts to how long transcription takes, so a CPU-only Mac isn't kept busy.
* If your mic picks up the other people through your speakers, those echoed lines are
  dropped (they're already in "Others").

Computer audio: Windows (WASAPI loopback, via the ``soundcard`` package) and macOS 13+
(ScreenCaptureKit, via the small ``murmr-audiotap`` helper built from mac/audiotap.swift;
macOS asks for "Screen & System Audio Recording" once). The mic works everywhere.
"""

from __future__ import annotations

import difflib
import json
import logging
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

log = logging.getLogger(__name__)

STEP_S = 1.5  # how often live text updates (at least)
MAX_LINE_S = 14.0  # a line becomes final after this long even without a pause
MIN_LINE_S = 2.0  # don't end a line before this much speech
PAUSE_S = 0.6  # a gap this long ends a line (comma pauses are shorter)
JOIN_GAP_S = 1.2  # a piece that continues the same sentence this soon joins the line before
SILENT_RMS = 0.004  # quieter than this (90th percentile): nobody is talking
ECHO_WINDOW_S = 15.0


AUDIOTAP = Path(__file__).resolve().parent / "resources" / "murmr-audiotap"


def _mac_version() -> tuple[int, ...]:
    import platform

    try:
        return tuple(int(x) for x in platform.mac_ver()[0].split(".") if x)
    except ValueError:
        return ()


def system_audio_supported() -> bool:
    if sys.platform == "darwin":
        return AUDIOTAP.exists() and _mac_version() >= (13,)
    if sys.platform != "win32":
        return False
    try:
        import soundcard  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


def SystemCapture() -> "Capture":  # noqa: N802 - picks the class for this OS
    return MacSystemCapture() if sys.platform == "darwin" else WindowsSystemCapture()


class SystemAudioDenied(RuntimeError):
    """macOS hasn't allowed Murmr to record the computer's audio."""


# --------------------------------------------------------------------------- audio sources
class _Buffer:
    """Audio from an absolute sample position on; consumed audio is dropped."""

    def __init__(self):
        self._chunks: list[np.ndarray] = []
        self._start = 0  # absolute index of the first kept sample
        self._end = 0
        self._lock = threading.Lock()

    def append(self, a: np.ndarray) -> None:
        with self._lock:
            self._chunks.append(a)
            self._end += len(a)

    @property
    def end(self) -> int:
        return self._end

    def take(self, start: int) -> np.ndarray:
        with self._lock:
            if not self._chunks:
                return np.zeros(0, np.float32)
            audio = np.concatenate(self._chunks) if len(self._chunks) > 1 else self._chunks[0]
            self._chunks = [audio]
        return audio[max(0, start - self._start):]

    def drop_before(self, pos: int) -> None:
        with self._lock:
            if not self._chunks or pos <= self._start:
                return
            audio = np.concatenate(self._chunks)
            cut = min(len(audio), pos - self._start)
            self._chunks = [audio[cut:]] if cut < len(audio) else []
            self._start += cut


class Capture:
    """An audio source: ``start()``, ``stop()``, and a buffer at ``samplerate``."""

    who = "You"

    def __init__(self):
        self.buffer = _Buffer()
        self.samplerate = 16000
        self.level = 0.0

    def _feed(self, mono: np.ndarray) -> None:
        mono = np.asarray(mono, np.float32)
        self.level = float(np.sqrt(np.mean(mono * mono))) if len(mono) else 0.0
        self.buffer.append(mono)

    def start(self) -> None: ...

    def stop(self) -> None: ...


class MicCapture(Capture):
    who = "You"

    def __init__(self, device=None):
        super().__init__()
        self.device = device
        self._stream = None

    def start(self) -> None:
        import sounddevice as sd

        from .audio import PA_LOCK, resolve_device

        with PA_LOCK:
            dev = resolve_device(self.device)
            self.samplerate = int(sd.query_devices(dev, "input")["default_samplerate"])
            self._stream = sd.InputStream(samplerate=self.samplerate, channels=1, dtype="float32",
                                          device=dev, callback=lambda d, f, t, s: self._feed(d[:, 0].copy()))
            self._stream.start()

    def stop(self) -> None:
        from .audio import PA_LOCK

        if self._stream is not None:
            with PA_LOCK:
                try:
                    self._stream.stop()
                finally:
                    self._stream.close()
            self._stream = None


class WindowsSystemCapture(Capture):
    """What the computer is playing (Windows WASAPI loopback of the default output)."""

    who = "Others"

    def __init__(self):
        super().__init__()
        self.samplerate = 48000
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.error = ""

    def start(self) -> None:
        import soundcard as sc

        speaker = sc.default_speaker()
        self._mic = sc.get_microphone(id=str(speaker.name), include_loopback=True)
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="system-audio", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        import warnings

        block = self.samplerate // 10
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # "data discontinuity" while nothing plays
                with self._mic.recorder(samplerate=self.samplerate, channels=1, blocksize=block) as r:
                    while not self._stop.is_set():
                        self._feed(r.record(numframes=block)[:, 0])
        except Exception as exc:  # noqa: BLE001 - output device changed, unplugged...
            self.error = str(exc)
            log.warning("Computer audio capture stopped: %s", exc)

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)


class MacSystemCapture(Capture):
    """What the Mac is playing, from the murmr-audiotap helper (float32 mono on stdout)."""

    who = "Others"

    def __init__(self, helper: Path | None = None, start_timeout: float = 8.0):
        super().__init__()
        self.helper = helper or AUDIOTAP
        self.samplerate = 48000
        self.start_timeout = start_timeout
        self.error = ""
        self._proc = None
        self._threads: list[threading.Thread] = []

    def _command(self) -> list[str]:
        import os

        return [str(self.helper), "--rate", str(self.samplerate), "--exclude-pid", str(os.getpid())]

    def start(self) -> None:
        import os
        import subprocess

        if not os.access(self.helper, os.X_OK):  # an update may not keep the executable bit
            try:
                os.chmod(self.helper, 0o755)
            except OSError:
                pass
        self._proc = subprocess.Popen(self._command(), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE, bufsize=0)
        started = threading.Event()
        state = {"line": "", "started": False}

        def errors():
            for raw in iter(self._proc.stderr.readline, b""):
                line = raw.decode("utf-8", "replace").strip()
                if line == "started":
                    state["started"] = True
                    started.set()
                elif line.startswith("rate "):
                    try:
                        self.samplerate = int(line.split()[1])
                    except ValueError:
                        pass
                elif line:
                    state["line"] = line
                    self.error = line.split(":", 1)[-1].strip()
                    log.warning("Computer audio helper: %s", line)
            started.set()  # it exited

        def reader():
            out = self._proc.stdout
            pending = b""
            while True:
                chunk = out.read(19200)  # 0.1 s at 48 kHz
                if not chunk:
                    break
                pending += chunk
                usable = len(pending) - len(pending) % 4
                if usable:
                    self._feed(np.frombuffer(pending[:usable], np.float32).copy())
                    pending = pending[usable:]

        self._threads = [threading.Thread(target=errors, name="system-audio-log", daemon=True),
                         threading.Thread(target=reader, name="system-audio", daemon=True)]
        for t in self._threads:
            t.start()
        started.wait(self.start_timeout)
        if not state["started"]:
            import subprocess

            try:
                code = self._proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.stop()
                raise RuntimeError("The computer audio didn't start (timed out)") from None
            for t in self._threads:
                t.join(timeout=1)
            self._proc = None
            if code == 2 or state["line"].startswith("denied"):
                raise SystemAudioDenied(
                    "Allow Murmr in System Settings > Privacy & Security > Screen & System Audio "
                    "Recording, then restart Murmr. Only the sound is used, never the screen.")
            raise RuntimeError(self.error or f"The computer audio helper stopped ({code})")

    def stop(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            proc.stdin.close()  # the helper exits when its stdin closes
        except OSError:
            pass
        try:
            proc.wait(timeout=2)
        except Exception:  # noqa: BLE001
            proc.kill()
        for t in self._threads:
            t.join(timeout=1)


def mac_system_audio_allowed() -> bool | None:
    """Screen & System Audio Recording allowed for Murmr? None: can't tell."""
    import subprocess

    if sys.platform != "darwin" or not AUDIOTAP.exists():
        return None
    try:
        r = subprocess.run([str(AUDIOTAP), "--check"], capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    return {"granted": True, "denied": False}.get(r.stdout.strip())


def request_mac_system_audio() -> None:
    """macOS's prompt (once), then System Settings at the right list."""
    import subprocess

    try:
        subprocess.run([str(AUDIOTAP), "--request"], capture_output=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        pass
    subprocess.Popen(["/usr/bin/open", "x-apple.systempreferences:com.apple.preference.security?"
                                       "Privacy_ScreenCapture"])


# --------------------------------------------------------------------------- segmentation
def _frames(audio: np.ndarray, sr: int, win_s: float = 0.05) -> np.ndarray:
    win = max(1, int(sr * win_s))
    n = len(audio) // win
    if n == 0:
        return np.zeros(0)
    return np.sqrt(np.mean(audio[: n * win].reshape(n, win) ** 2, axis=1))


def is_silent(audio: np.ndarray, sr: int) -> bool:
    rms = _frames(audio, sr)
    return not len(rms) or float(np.percentile(rms, 90)) < SILENT_RMS


def find_pause(audio: np.ndarray, sr: int) -> int | None:
    """Sample index in the middle of the last pause that follows enough speech, or None."""
    rms = _frames(audio, sr)
    if len(rms) < 4:
        return None
    loud = float(np.percentile(rms, 90))
    quiet = rms < max(SILENT_RMS, 0.12 * loud)
    need = max(1, int(PAUSE_S / 0.05))
    win = int(sr * 0.05)
    best, run = None, 0
    for i, q in enumerate(quiet):
        run = run + 1 if q else 0
        start = i - run + 1
        if run >= need and start * 0.05 >= MIN_LINE_S and i < len(quiet) - 1:
            best = (start + run // 2) * win  # middle of the pause so far
    return best


# --------------------------------------------------------------------------- session
@dataclass
class Channel:
    capture: Capture
    committed: int = 0  # absolute sample: audio before this is final
    partial: str = ""
    partial_at: int = 0  # buffer end when the partial was made


@dataclass
class Line:
    t: float  # seconds from the start
    who: str  # the speaker: "You", "Person 2", or the name you gave them
    text: str
    end: float = 0.0
    channel: str = ""  # the source: "You" (mic) or "Others" (computer audio)
    edited: bool = False  # corrected by hand during the meeting


@dataclass
class MeetingSession:
    """A running transcription. ``keep=False``: live captions only (nothing saved)."""

    captures: list[Capture]
    transcribe: Callable[[np.ndarray, int], str]
    on_update: Callable[[], None] = lambda: None
    keep: bool = True
    embedder: Callable[[np.ndarray, int], np.ndarray | None] | None = None  # voiceprints
    profile_dir: Path | None = None  # your recordings, to recognise your voice (mic-only)
    room: bool = False  # several people share the microphone (a meeting room)
    lines: list[Line] = field(default_factory=list)
    started: float = field(default_factory=time.time)

    def __post_init__(self):
        self.channels = [Channel(c) for c in self.captures]
        self._trackers: dict = {}  # channel -> speakers.Tracker (set up at start)
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self.running = False
        self.problems: list[tuple[str, Exception]] = []

    # -- control ---------------------------------------------------------
    def start(self) -> None:
        started = []
        self.problems = []  # sources that couldn't start, to tell you why
        for c in self.captures:
            try:
                c.start()
                started.append(c)
            except Exception as exc:  # noqa: BLE001
                log.warning("Couldn't start %s audio: %s", c.who, exc)
                self.problems.append((c.who, exc))
        if not started:
            raise RuntimeError(str(self.problems[0][1]) if self.problems
                               else "No audio source could be started")
        self.channels = [Channel(c) for c in started]
        self._trackers = {}  # set up by the meeting thread: loading voiceprints takes seconds
        self.started = time.time()
        self.running = True
        self._thread = threading.Thread(target=self._run, name="meeting", daemon=True)
        self._thread.start()

    def stop(self) -> list[Line]:
        """Stop capturing and finish the last lines. Blocks until done."""
        self._stop.set()
        for ch in self.channels:
            ch.capture.stop()
        if self._thread:
            self._thread.join(timeout=60)
        for ch in self.channels:
            self._finish(ch)
        self.running = False
        self.on_update()
        return self.lines

    @property
    def elapsed(self) -> float:
        return time.time() - self.started

    def live(self) -> list[tuple[str, str]]:
        """(who, text) of the lines still being spoken."""
        return [(self._trackers.get(ch.capture.who).last if self._trackers.get(ch.capture.who) and
                 self._trackers[ch.capture.who].last else ch.capture.who, ch.partial)
                for ch in self.channels if ch.partial]

    # -- your corrections while it runs ---------------------------------------
    def rename(self, old: str, new: str) -> None:
        """Name a speaker now: their lines so far, and new ones from that voice."""
        new = " ".join(new.split())
        if not new or new == old:
            return
        with self._lock:
            for line in self.lines:
                if line.who == old:
                    line.who = new
            for tracker in self._trackers.values():
                tracker.rename(old, new)
        self.on_update()

    def edit_line(self, index: int, text: str) -> str | None:
        """Correct a line's text. Returns what it said before (or None if it's gone)."""
        text = " ".join(text.split())
        with self._lock:
            if not 0 <= index < len(self.lines) or not text:
                return None
            before = self.lines[index].text
            self.lines[index].text, self.lines[index].edited = text, True
        self.on_update()
        return before

    # -- who's talking -----------------------------------------------------
    def _setup_speakers(self) -> None:
        """Tell voices apart in the computer's audio, and in the mic when it's the only
        source (a room). Your own voice is recognised there from your recordings."""
        from .speakers import Tracker

        self._trackers: dict = {}
        if self.embedder is None:
            return
        whos = [ch.capture.who for ch in self.channels]
        counter = [1]  # one numbering for the room and the call: never two "Person 1"s
        if "Others" in whos:
            self._trackers["Others"] = Tracker("Person", counter=counter)
        if whos == ["You"] or (self.room and "You" in whos):
            known = {}
            if self.profile_dir is not None:
                from .speakers import voice_profile

                try:
                    me = voice_profile(self.embedder, self.profile_dir)
                    if me is not None:
                        known["You"] = me
                except Exception:  # noqa: BLE001
                    log.debug("No voice profile", exc_info=True)
            self._trackers["You"] = Tracker("Person", known=known, counter=counter)

    def _speaker(self, ch: Channel, audio: np.ndarray) -> str:
        tracker = self._trackers.get(ch.capture.who) if hasattr(self, "_trackers") else None
        if tracker is None:
            return ch.capture.who
        try:
            emb = self.embedder(audio, ch.capture.samplerate)
        except Exception:  # noqa: BLE001
            emb = None
        return tracker.assign(emb, "You" if ch.capture.who == "You" and tracker.known else ch.capture.who)

    # -- work ------------------------------------------------------------
    def _run(self) -> None:
        try:
            self._setup_speakers()
        except Exception:  # noqa: BLE001 - transcribe anyway, without telling voices apart
            log.exception("Couldn't set up telling speakers apart")
        while not self._stop.is_set():
            t0 = time.monotonic()
            for ch in self.channels:
                if self._stop.is_set():
                    break
                try:
                    if self._step(ch):
                        self.on_update()
                except Exception:  # noqa: BLE001 - keep going: one bad step mustn't end the notes
                    log.exception("Meeting transcription step failed")
            busy = time.monotonic() - t0
            # Stay light: never more than about a third of the time transcribing.
            self._stop.wait(max(STEP_S, busy * 2))

    def _step(self, ch: Channel) -> bool:
        cap = ch.capture
        sr = cap.samplerate
        audio = cap.buffer.take(ch.committed)
        n = len(audio)
        if n < 0.4 * sr:
            return False
        if is_silent(audio, sr):
            if n > 1.5 * sr:  # nobody talking: forget the silence (keep a little lead-in)
                had_partial = bool(ch.partial)
                if ch.partial:
                    self._add(ch, ch.committed, ch.partial)
                ch.partial = ""
                self._commit(ch, n - int(0.3 * sr))
                return had_partial
            return False
        cut = find_pause(audio, sr) if n >= (MIN_LINE_S + PAUSE_S) * sr else None
        if cut is None and n >= MAX_LINE_S * sr:
            from .streaming import find_cut

            cut = find_cut(audio, sr, force=True)
        if cut:
            text = self.transcribe(audio[:cut], sr).strip()
            self._add(ch, ch.committed, text, cut, self._speaker(ch, audio[:cut]) if text else "")
            ch.partial = ""
            self._commit(ch, cut)
            return True
        if cap.buffer.end - ch.partial_at < 0.8 * sr:
            return False  # not enough new audio to bother
        ch.partial = self.transcribe(audio, sr).strip()
        ch.partial_at = cap.buffer.end
        return True

    def _finish(self, ch: Channel) -> None:
        sr = ch.capture.samplerate
        audio = ch.capture.buffer.take(ch.committed)
        if len(audio) >= 0.4 * sr and not is_silent(audio, sr):
            try:
                self._add(ch, ch.committed, self.transcribe(audio, sr).strip(), len(audio),
                          self._speaker(ch, audio))
            except Exception:  # noqa: BLE001
                log.exception("Last meeting line failed")
        elif ch.partial:
            self._add(ch, ch.committed, ch.partial)
        ch.partial = ""

    def _commit(self, ch: Channel, n: int) -> None:
        ch.committed += n
        ch.capture.buffer.drop_before(ch.committed)

    def _add(self, ch: Channel, start_sample: int, text: str, length: int = 0, speaker: str = "") -> None:
        if not text:
            return
        sr = ch.capture.samplerate
        t, end = start_sample / sr, (start_sample + length) / sr
        channel = ch.capture.who
        who = speaker or channel
        with self._lock:
            if self._is_echo(channel, t, text):
                log.debug("Dropped echoed line: %s", text)
                return
            last = next((line for line in reversed(self.lines) if (line.channel or line.who) == channel), None)
            if last is not None and last.who != who:
                last = None  # a different person: a new line
            if (last is not None and last is self.lines[-1] and t - last.end <= JOIN_GAP_S
                    and not last.text.rstrip().endswith((".", "?", "!"))):
                # The same sentence, split at a short pause: continue the line.
                last.text = f"{last.text.rstrip()} {text[0].lower() + text[1:] if text[1:2].islower() else text}"
                last.end = round(end, 1)
                return
            if last is not None and last is self.lines[-1] and t - last.end <= JOIN_GAP_S \
                    and last.text.rstrip().endswith(".") and text[:1].islower():
                last.text = last.text.rstrip()[:-1] + " " + text  # "ship on the 20th." + "if ..."
                last.end = round(end, 1)
                return
            self.lines.append(Line(round(t, 1), who, text, round(end, 1), channel))
            self.lines.sort(key=lambda line: line.t)

    def _is_echo(self, who: str, t: float, text: str) -> bool:
        """Your mic picking up the other people: drop that copy (or replace a mic copy that
        arrived first with the computer-audio original). ``who`` is the channel."""
        if len({c.capture.who for c in self.channels}) < 2:
            return False
        key = text.lower()
        for line in reversed(self.lines):
            if abs(line.t - t) > ECHO_WINDOW_S:
                continue
            if (line.channel or line.who) == who:
                continue
            if difflib.SequenceMatcher(None, key, line.text.lower()).ratio() >= 0.6:
                if who == "You":
                    return True  # this is the echo
                self.lines.remove(line)  # the earlier "You" line was the echo
                return False
        return False


# --------------------------------------------------------------------------- text out
def fmt_time(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600}:{s // 60 % 60:02d}:{s % 60:02d}" if s >= 3600 else f"{s // 60}:{s % 60:02d}"


def transcript_text(lines: list[dict] | list[Line]) -> str:
    out = []
    for line in lines:
        d = line if isinstance(line, dict) else line.__dict__
        out.append(f"[{fmt_time(d['t'])}] {d['who']}: {d['text']}")
    return "\n".join(out)


def to_markdown(meeting: dict) -> str:
    when = time.strftime("%A %d %B %Y, %H:%M", time.localtime(meeting["started"]))
    parts = [f"# {meeting.get('title') or 'Meeting notes'}", "",
             f"{when} · {fmt_time(meeting.get('duration', 0))}", ""]
    if meeting.get("summary"):
        parts += [meeting["summary"].strip(), ""]
    parts += ["## Transcript", "", transcript_text(meeting.get("lines", []))]
    return "\n".join(parts) + "\n"


# --------------------------------------------------------------------------- storage
class MeetingStore:
    """One JSON file per meeting in <config>/meetings. Text only, no audio."""

    def __init__(self, root: Path):
        self.root = root

    def save(self, meeting: dict) -> str:
        self.root.mkdir(parents=True, exist_ok=True)
        meeting.setdefault("id", time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6])
        tmp = self.root / f"{meeting['id']}.json.tmp"
        tmp.write_text(json.dumps(meeting, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(self.root / f"{meeting['id']}.json")
        return meeting["id"]

    def load(self, meeting_id: str) -> dict | None:
        try:
            return json.loads((self.root / f"{meeting_id}.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def list(self) -> list[dict]:
        out = []
        for p in sorted(self.root.glob("*.json"), reverse=True) if self.root.exists() else []:
            try:
                m = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            out.append({k: m.get(k) for k in ("id", "title", "started", "duration", "summary")}
                       | {"words": sum(len(line["text"].split()) for line in m.get("lines", []))})
        return out

    def delete(self, meeting_id: str) -> None:
        (self.root / f"{meeting_id}.json").unlink(missing_ok=True)

    def delete_all(self) -> None:
        for p in self.root.glob("*.json") if self.root.exists() else []:
            p.unlink(missing_ok=True)
