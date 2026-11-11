import pytest

from whispa.speak import split_sentences


def test_split_sentences():
    text = "Hi Sam! See https://example.com/x?y=1 for details.\n\n- First point\n- Second point"
    assert split_sentences(text) == ["Hi Sam!", "See link for details.", "First point", "Second point"]


def test_espeak_phonemes_keep_punctuation():
    pytest.importorskip("espeakng_loader")
    from whispa.speak import Espeak

    ps = Espeak().phonemize("Hello, world!")
    assert ps.startswith("h") and "," in ps and ps.endswith("!")
    assert "ˈ" in ps  # stress marks: Kokoro needs them


def test_first_use_speaks_with_the_system_voice_while_the_natural_voice_downloads(monkeypatch):
    import threading
    import time

    import whispa.speak as sp

    states, spoken = [], []
    loaded = threading.Event()

    class FakeKokoro:
        def __init__(self, device):
            time.sleep(0.2)  # "downloading"

        def voice(self, name):
            loaded.set()

    monkeypatch.setattr(sp, "Kokoro", FakeKokoro)
    monkeypatch.setattr(sp, "voice_downloaded", lambda voice, device: False)
    monkeypatch.setattr(sp.Speaker, "_run_system",
                        lambda self, text, gen, announce=True: spoken.append((text, announce)))
    s = sp.Speaker(on_state=states.append)
    s.speak("Hello there.")
    assert loaded.wait(3)
    for _ in range(50):
        if "ready" in states and "idle" in states:
            break
        time.sleep(0.05)
    assert spoken == [("Hello there.", False)]  # right away, no silent wait
    assert states[0] == "downloading" and "ready" in states
    assert s._kokoro is not None and not s.speaking


def test_my_voices_are_listed_first(tmp_path, monkeypatch):
    import json

    from whispa import config as config_mod
    from whispa import speak

    monkeypatch.setattr(config_mod, "config_dir", lambda: tmp_path)
    assert speak.all_voices() == speak.VOICES  # nothing trained yet
    v = tmp_path / "voices" / "me"
    v.mkdir(parents=True)
    (v / "voice.onnx").write_bytes(b"x")
    (v / "voice.onnx.json").write_text("{}")
    (v / "info.json").write_text(json.dumps({"label": "My voice"}))
    (tmp_path / "voices" / "half-copied").mkdir()  # no model files: not listed
    assert speak.all_voices()[0] == ("my:me", "My voice · from your recordings")
    assert len(speak.all_voices()) == len(speak.VOICES) + 1
