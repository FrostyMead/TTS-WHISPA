"""Never lose a transcript: whatever fails after speech recognition, you get your words.
And Insert into the original field never pastes somewhere it can't confirm."""
import threading

import numpy as np
import pytest

from whispa import app as app_mod
from whispa import focus
from whispa.config import Config, LLMCleanup
from whispa.llm import Polisher


class Sig:
    def __init__(self):
        self.calls = []

    def emit(self, *args):
        self.calls.append(args)


class Bridge:
    def __init__(self):
        for name in ("done", "failed", "notice", "show_card", "history_changed"):
            setattr(self, name, Sig())


def make_app(raw="hello there", clean=None, deliver=None):
    a = app_mod.MurmrApp.__new__(app_mod.MurmrApp)
    a.cfg = Config()
    a.bridge = Bridge()
    ready = threading.Event()
    ready.set()
    a.transcriber = type("T", (), {"ready": ready, "backend": "CPU",
                                   "transcribe": staticmethod(lambda audio, sr: raw)})()
    a.polisher = Polisher(LLMCleanup(enabled=False))
    a.training = None
    a.remembered, a.delivered, a.copied = [], [], []
    a._remember = lambda raw, text, secs, command=None: a.remembered.append((raw, text))
    a._deliver = deliver or (lambda text: a.delivered.append(text))
    a._clean = clean or (lambda raw: raw.capitalize() + ".")
    a._keep_sample = lambda: False
    a._watch_edits = lambda text, sample: None
    a.injector = type("I", (), {"copy": staticmethod(lambda t: a.copied.append(t))})()
    a._focus = focus.Focus()
    return a


AUDIO = np.full(16000, 0.2, np.float32)


def test_normal_dictation_is_remembered_and_pasted():
    a = make_app()
    a._process(AUDIO, 16000)
    assert a.delivered == ["Hello there."] and a.remembered == [("hello there", "Hello there.")]
    assert a.bridge.done.calls == [("Hello there.",)]


def test_tidying_fails_you_get_the_words_as_heard():
    def boom(raw):
        raise RuntimeError("textproc exploded")

    a = make_app(clean=boom)
    a._process(AUDIO, 16000)
    assert a.delivered == ["hello there"] and a.remembered == [("hello there", "hello there")]
    assert any("as heard" in c[0] for c in a.bridge.notice.calls)


def test_paste_fails_the_text_is_kept_and_shown_to_copy():
    def broken(text):
        raise OSError("clipboard locked")

    a = make_app(deliver=broken)
    a._process(AUDIO, 16000)
    assert a.remembered == [("hello there", "Hello there.")]  # in history / Paste last
    assert a.copied == ["Hello there."]
    text, why = a.bridge.show_card.calls[0]
    assert text == "Hello there." and "Couldn't paste" in why
    assert a.bridge.done.calls == [("Hello there.",)]


def test_silence_says_check_the_microphone():
    a = make_app(raw="")
    a._process(np.zeros(16000, np.float32), 16000)
    assert a._empty_reason.startswith("No sound") and a.bridge.done.calls == [("",)]


def test_slow_ai_clean_up_is_reported(monkeypatch):
    p = Polisher(LLMCleanup(enabled=True, when="always"))
    monkeypatch.setattr(p, "active", lambda wait=False: True)

    def slow(*a, **k):
        raise TimeoutError("timed out")

    monkeypatch.setattr(p, "_complete", slow)
    assert p.polish("so um hi", app="") == "so um hi" and p.last_issue == "timeout"


def test_original_field_gone_shows_the_copy_card(monkeypatch):
    a = make_app()
    a.cfg.insert_original = True
    start = focus.Focus(is_text=True, app="Outlook", _handle=object(), _same=lambda other: False)
    now = focus.Focus(is_text=True, app="Slack", _handle=object(), _same=lambda other: False)
    a._cur_target = _done(start)
    a._card_reason = ""
    monkeypatch.setattr(focus, "inspect", lambda: now)
    monkeypatch.setattr(focus, "restore", lambda target: False)
    assert a._target() is start and "Outlook" in a._card_reason


def test_original_field_brought_back(monkeypatch):
    a = make_app()
    a.cfg.insert_original = True
    start = focus.Focus(is_text=True, app="Outlook", _handle=object())
    now = focus.Focus(is_text=True, app="Slack", _handle=object(), _same=lambda other: False)
    back = focus.Focus(is_text=True, app="Outlook", _handle=start._handle)
    looks = iter([now, back])
    a._cur_target, a._card_reason = _done(start), ""
    monkeypatch.setattr(focus, "inspect", lambda: next(looks))
    monkeypatch.setattr(focus, "restore", lambda target: True)
    assert a._target() is back and a._card_reason == ""


def test_option_off_uses_whatever_is_focused(monkeypatch):
    a = make_app()
    now = focus.Focus(is_text=True, app="Slack")
    a._cur_target = _done(focus.Focus(app="Outlook", _handle=object()))
    monkeypatch.setattr(focus, "inspect", lambda: now)
    assert a._target() is now


def _done(f):
    ev = threading.Event()
    ev.set()
    return {"focus": f, "done": ev}


def test_unknown_start_field_is_never_guessed(monkeypatch):
    a = make_app()
    a.cfg.insert_original = True
    a._cur_target, a._card_reason = _done(None), ""  # reading the start failed
    monkeypatch.setattr(focus, "inspect", lambda: focus.Focus(is_text=True, app="Slack", _handle=object()))
    a._target()
    assert "Couldn't tell" in a._card_reason


def test_each_recording_keeps_its_own_start_field():
    a = make_app()
    first, second = _done(focus.Focus(app="Outlook")), _done(focus.Focus(app="Slack"))
    a._process(AUDIO, 16000, None, first)
    assert a._cur_target is first
    a._process(AUDIO, 16000, None, second)
    assert a._cur_target is second


def test_history_save_failing_still_pastes():
    a = make_app()

    def broken(*args, **kw):
        raise OSError("disk full")

    a._remember = broken
    a._process(AUDIO, 16000)
    assert a.delivered == ["Hello there."] and a.bridge.done.calls == [("Hello there.",)]


def test_restore_without_a_handle_never_claims_success():
    assert focus.restore(focus.Focus(app="x")) is False
