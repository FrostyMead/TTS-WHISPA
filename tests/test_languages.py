"""Issue #4: Chinese (and other non-European) dictation and read aloud, and saying why
Apple's on-device model isn't used on an older macOS."""
import subprocess
import threading
import types

import numpy as np

from whispa import asr, speak


# --- recognition -------------------------------------------------------------------------
def test_language_choices():
    assert asr.LANGUAGE_MODELS["european"] == asr.BASE_MODEL
    assert asr.is_whisper(asr.LANGUAGE_MODELS["world"]) and asr.is_whisper(asr.LANGUAGE_MODELS["world_best"])
    assert not asr.is_whisper(asr.BASE_MODEL)


def test_whisper_always_uses_the_compact_model():
    t = asr.Transcriber(asr.WHISPER_MODEL, "cpu", "none")
    assert t._quant("CPU") == "int8" and t._quant("CUDA") == "int8"
    assert asr.Transcriber(asr.BASE_MODEL, "cpu", "none")._quant("CPU") is None


def test_pieces_join_without_spaces_in_chinese():
    assert asr.join_pieces(["今天天气很好。", "我们去公园吧。"]) == "今天天气很好。我们去公园吧。"
    assert asr.join_pieces(["Hello there.", "How are you?"]) == "Hello there. How are you?"
    assert asr.join_pieces(["我在用Murmr", "it works"]) == "我在用Murmr it works"


def test_whisper_ghost_text_is_dropped():
    assert asr.drop_ghosts("字幕由Amara.org社区提供") == ""
    assert asr.drop_ghosts("我们明天见。") == "我们明天见。"


class _Model:
    def __init__(self, text):
        self.text, self.calls = text, 0

    def recognize(self, audio, sample_rate):
        self.calls += 1
        return self.text


def _ready(model_name, text):
    t = asr.Transcriber(model_name, "cpu")
    t._model = _Model(text)
    t.ready.set()
    return t


def test_whisper_hears_nothing_in_silence():
    t = _ready(asr.WHISPER_MODEL, "Thank you.")
    assert t.transcribe(np.zeros(16000, np.float32), 16000) == ""
    assert t._model.calls == 0
    loud = (np.sin(np.arange(16000) / 5) * 0.2).astype(np.float32)
    assert t.transcribe(loud, 16000) == "Thank you."


def test_parakeet_is_unchanged_by_the_silence_rule():
    t = _ready(asr.BASE_MODEL, "")
    t.transcribe(np.zeros(16000, np.float32), 16000)
    assert t._model.calls == 1


def test_whisper_splits_after_28_seconds(monkeypatch):
    t = _ready(asr.WHISPER_MODEL, "x")
    called = []
    monkeypatch.setattr(t, "_transcribe_long", lambda a, sr: called.append(len(a) / sr) or "long")
    loud = (np.sin(np.arange(16000 * 29) / 5) * 0.2).astype(np.float32)
    assert t.transcribe(loud, 16000) == "long" and called  # Parakeet would wait until 60 s


# --- read aloud --------------------------------------------------------------------------
def test_chinese_text_is_detected():
    assert speak.mostly_cjk("今天天气很好，我们去公园吧。") == "zh"
    assert speak.mostly_cjk("これは日本語のテキストです。") == "ja"
    assert speak.mostly_cjk("안녕하세요, 반갑습니다.") == "ko"
    assert speak.mostly_cjk("Please read this English sentence.") is None
    assert speak.mostly_cjk("I like 中文 a lot, honestly, it's great") is None  # mostly English


def test_mac_chinese_voice_is_found(monkeypatch):
    out = ("Albert              en_US    # Hello! My name is Albert.\n"
           "Meijia              zh_TW    # 你好，我叫美佳。\n"
           "Tingting            zh_CN    # 你好，我叫婷婷。\n")
    monkeypatch.setattr(speak.subprocess, "run",
                        lambda *a, **k: types.SimpleNamespace(stdout=out, returncode=0))
    speak._MAC_VOICES.clear()
    assert speak.mac_voice_for("zh") == "Tingting"  # mainland first
    assert speak.mac_voice_for("ko") is None
    speak._MAC_VOICES.clear()


def test_chinese_is_read_by_the_system_voice(monkeypatch):
    s = speak.Speaker(voice="af_heart")
    used = []
    monkeypatch.setattr(s, "_run_system", lambda text, gen, announce=True, lang=None: used.append(lang))
    monkeypatch.setattr(s, "_run_kokoro", lambda text, gen: used.append("kokoro"))
    s._run("你好，世界。今天很好。", s._gen)
    assert used == ["zh"]


# --- Apple's on-device model -------------------------------------------------------------
def test_older_macos_says_why_apple_model_is_not_used(monkeypatch):
    import platform

    from whispa import ai_setup, applellm

    monkeypatch.setattr(ai_setup.sys, "platform", "darwin")
    monkeypatch.setattr(platform, "machine", lambda: "arm64")
    monkeypatch.setattr(applellm, "supported", lambda: False)
    monkeypatch.setattr(applellm, "os_version", lambda: "15.2")
    why = ai_setup.apple_too_old()
    assert "macOS 26" in why and "15.2" in why
    monkeypatch.setattr(applellm, "supported", lambda: True)
    assert ai_setup.apple_too_old() == ""


# --- Settings ----------------------------------------------------------------------------
from tests.test_responsive import window  # noqa: E402,F401


def test_settings_offers_the_languages(window):  # noqa: F811
    app, w = window
    combo = w.settings.languages
    assert [combo.itemData(i) for i in range(combo.count())] == ["european", "world", "world_best"]
    asked = []
    w.settings.ctl.set_speech_languages = asked.append
    combo.setCurrentIndex(1)
    assert asked == ["world"]
