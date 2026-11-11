"""Per-app styles, names from the screen, live transcription, streak, action shortcuts."""

import datetime as dt
import threading

import numpy as np
import pytest

from whispa import styles, textproc
from whispa.audio import Recorder
from whispa.hotkey import Hotkey, HotkeyService, PTTStateMachine
from whispa.streaming import LiveSession, find_cut


# --------------------------------------------------------------------------- styles
@pytest.mark.parametrize("app,title,terminal,cat", [
    ("slack", "", False, "chat"),
    ("Slack", "", False, "chat"),
    ("OUTLOOK", "", False, "email"),
    ("olk", "", False, "email"),
    ("Code", "", False, "code"),
    ("WindowsTerminal", "", False, "code"),
    ("weird-shell", "", True, "code"),
    ("chrome", "Inbox (3) - me@example.com - Gmail", False, "email"),
    ("msedge", "general | Acme - Slack", False, "chat"),
    ("chrome", "Kubernetes docs", False, "other"),
    ("winword", "Report.docx", False, "other"),
])
def test_category(app, title, terminal, cat):
    assert styles.category(app, title, terminal) == cat


def test_style_for_uses_defaults_and_ignores_unknown_values():
    assert styles.style_for({}, "slack") == "casual"
    assert styles.style_for({"chat": "nonsense"}, "slack") == "casual"
    assert styles.style_for({"code": "formal"}, "cursor") == "formal"


def test_casual_drops_the_final_full_stop_only():
    assert styles.apply("Sounds good. See you at five.", "casual") == "Sounds good. See you at five"
    assert styles.apply("Are you coming?", "casual") == "Are you coming?"
    assert styles.apply("Well...", "casual") == "Well..."
    assert styles.apply("Line one.\nLine two.", "casual") == "Line one.\nLine two."
    assert styles.apply("Sounds good.", "formal") == "Sounds good."


def test_very_casual_lowercases_sentence_starts_but_keeps_names():
    out = styles.apply("Sounds good. I'll ask Siobhan. NASA called. Kubernetes is down.",
                       "very_casual", keep={"Kubernetes"})
    assert out == "sounds good. I'll ask Siobhan. NASA called. Kubernetes is down"
    assert styles.apply("iPhone is here.", "very_casual") == "iPhone is here"


# --------------------------------------------------------------------------- screen names
def test_candidate_names():
    text = ("Re: Budget for Q3\nHi team, Siobhan and Aoife reviewed the useEffect hook. "
            "Siobhan thinks GPT4o is fine. The meeting is on Monday.")
    names = textproc.candidate_names(text)
    assert names[0] == "Siobhan"
    assert {"Aoife", "useEffect", "GPT4o"} <= set(names)
    assert "Monday" not in names and "The" not in names and "Hi" not in names


def test_careful_matching_only_fixes_misheard_names():
    names = ["Siobhan", "Gather", "Kowalczyk"]
    fix = lambda t: textproc.apply_terms(t, names, threshold=0.86, min_len=5, case_only=False)
    assert fix("Ask Kowalcyk about it") == "Ask Kowalczyk about it"
    assert fix("she gathers the notes") == "she gathers the notes"  # not heard as a name
    assert fix("Gathers is a word") == "Gathers is a word"  # just a suffix
    assert fix("ask kowalcyk") == "ask kowalcyk"  # lower case: not heard as a name


# --------------------------------------------------------------------------- live transcription
def _speech(seconds, sr=16000, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(int(seconds * sr)) * 0.2).astype(np.float32)


def test_find_cut_lands_in_a_pause():
    sr = 16000
    audio = np.concatenate([_speech(20), np.zeros(sr // 2, np.float32), _speech(2)])
    cut = find_cut(audio, sr)
    assert 20 * sr <= cut <= 20.5 * sr
    assert find_cut(_speech(22), sr) is None  # no pause: wait
    assert find_cut(_speech(22), sr, force=True) is not None


def test_live_session_transcribes_pieces_and_the_rest():
    sr = 16000
    audio = np.concatenate([_speech(21), np.zeros(sr // 2, np.float32), _speech(5, seed=1)])
    seen = []

    def transcribe(a, rate):
        seen.append(len(a))
        return f"part{len(seen)}"

    live = LiveSession(transcribe)
    live.poll(lambda start: audio[start:], sr)
    assert live.parts and 21 * sr <= live.cut <= 21.5 * sr
    live.poll(lambda start: audio[start:], sr)  # not enough new audio yet
    assert len(live.parts) == 1
    assert live.finish(audio, sr) == "part1 part2"
    assert sum(seen) == len(audio)


def test_recorder_samples_since():
    rec = Recorder.__new__(Recorder)
    rec._lock = threading.Lock()
    rec._chunks = [np.arange(0, 100, dtype=np.float32), np.arange(100, 250, dtype=np.float32)]
    rec._samples = 250
    assert rec.samples_since(0).tolist() == list(range(250))
    assert rec.samples_since(120).tolist() == list(range(120, 250))
    assert rec.samples_since(250).size == 0


# --------------------------------------------------------------------------- streak
def test_day_streak():
    from whispa.window import day_streak

    today = dt.date(2026, 11, 11)
    d = lambda n: today - dt.timedelta(days=n)
    assert day_streak({d(0), d(1), d(2), d(4)}, today) == 3
    assert day_streak({d(1), d(2)}, today) == 2  # not yet today: streak still alive
    assert day_streak({d(2)}, today) == 0
    assert day_streak(set(), today) == 0


# --------------------------------------------------------------------------- action shortcuts
def _service():
    sm = PTTStateMachine(Hotkey.parse("ctrl+cmd"), on_start=lambda: None, on_stop=lambda: None,
                         on_cancel=lambda s: None, on_lock=lambda: None)
    return HotkeyService(sm)


def test_action_shortcut_fires_once_per_press():
    svc = _service()
    fired = []
    svc.set_action("paste_last", Hotkey.parse("alt+shift+z"), lambda: fired.append(1))
    svc._press({"alt", "alt_l"}, "alt_l")
    svc._press({"shift", "shift_l"}, "shift_l")
    assert not fired
    svc._press({"z"}, "z")
    svc._press({"z"}, "z")  # key repeat
    assert fired == [1]
    svc._release({"z"}, "z")
    svc._press({"z"}, "z")
    assert fired == [1, 1]


def test_action_shortcut_sharing_modifiers_with_push_to_talk():
    svc = _service()
    fired, started, cancelled = [], [], []
    svc.sm.on_start = lambda: started.append(1)
    svc.sm.on_cancel = lambda silent: cancelled.append(silent)
    svc.set_action("read", Hotkey.parse("ctrl+cmd+r"), lambda: fired.append(1))
    svc._press({"ctrl", "ctrl_l"}, "ctrl_l")
    svc._press({"cmd", "cmd_l"}, "cmd_l")  # push to talk starts...
    svc._press({"r"}, "r")  # ...but R arrives right away: it was the read shortcut
    assert started and cancelled == [True] and fired == [1]


def test_disabled_action_does_nothing():
    svc = _service()
    fired = []
    svc.set_action("read", Hotkey.parse("alt+shift+r"), lambda: fired.append(1))
    svc.set_action("read", None, lambda: fired.append(1))
    for names, p in (({"alt"}, "alt_l"), ({"shift"}, "shift_l"), ({"r"}, "r")):
        svc._press(names, p)
    assert not fired


# --------------------------------------------------------------------------- macOS focus
def test_mac_classify_pastes_when_unsure():
    from whispa.focus import mac_classify

    assert mac_classify("AXTextArea", "", False, False, False, "Notes") is True
    assert mac_classify("AXGroup", "", False, False, True, "Slack") is True  # has a caret
    assert mac_classify("AXWebArea", "", False, False, False, "Google Chrome") is None  # paste
    assert mac_classify("AXGroup", "", False, False, False, "Code") is None  # paste
    assert mac_classify("AXButton", "", False, False, False, "Safari") is False  # copy card
    assert mac_classify("AXList", "", False, False, False, "Finder") is False
    assert mac_classify("AXScrollArea", "", False, False, False, "Finder") is False  # desktop


def test_portaudio_calls_take_turns(monkeypatch):
    """Opening/closing streams from two threads at once can deadlock PortAudio (macOS)."""
    import threading
    import time

    from whispa import audio

    active, overlap = [0], []

    class FakeStream:
        def __init__(self, **kw):
            active[0] += 1
            if active[0] > 1:
                overlap.append(1)
            time.sleep(0.01)
            active[0] -= 1
            self.active = True

        def start(self):
            pass

        def stop(self):
            active[0] += 1
            if active[0] > 1:
                overlap.append(1)
            time.sleep(0.01)
            active[0] -= 1

        def close(self):
            pass

        def write(self, data):
            pass

    monkeypatch.setattr(audio.sd, "OutputStream", FakeStream)

    def play():
        for _ in range(5):
            with audio.output_stream(samplerate=16000) as out:
                out.write(b"")

    threads = [threading.Thread(target=play) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not overlap
