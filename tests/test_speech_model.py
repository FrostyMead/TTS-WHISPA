import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from whispa import asr  # noqa: E402
from whispa.config import Config  # noqa: E402
from whispa.speech_model import SpeechModelSection, installed_models  # noqa: E402

REPORT = {
    "recommend": True, "reason": "ok",
    "base": {"yours": {"wer": 0.084}, "general": {"wer": 0.060}},
    "tuned": {"yours": {"wer": 0.051}, "general": {"wer": 0.061}},
    "training": {"user_minutes": 42.3, "created": "2026-11-12 10:30"},
}


def make_model(root, name, report=REPORT, int8=False):
    d = root / name
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps({"model_type": "nemo-conformer-tdt"}))
    for f in ("encoder-model.onnx", "decoder_joint-model.onnx", "vocab.txt"):
        (d / f).write_text("x")
    if int8:
        (d / "encoder-model.int8.onnx").write_text("x")
    if report is not None:
        (d / "eval.json").write_text(json.dumps(report))
    return d


# --- asr: local folders ---------------------------------------------------------------------
def test_local_model_dir(tmp_path):
    d = make_model(tmp_path, "tuned-1")
    assert asr.local_model_dir(str(d)) == d
    assert asr.local_model_dir("nemo-parakeet-tdt-0.6b-v3") is None
    assert asr.local_model_dir(str(tmp_path / "missing")) is None


def test_local_folder_is_loaded_by_type_and_skips_missing_int8(tmp_path, monkeypatch):
    import onnx_asr

    calls = []
    monkeypatch.setattr(onnx_asr, "load_model", lambda *a, **k: calls.append((a, k)) or "model")
    d = make_model(tmp_path, "tuned-1")
    asr._load(str(d), "int8", ["CPUExecutionProvider"])
    assert calls[-1][0] == ("nemo-conformer-tdt", d) and calls[-1][1]["quantization"] is None
    make_model(tmp_path, "tuned-2", int8=True)
    asr._load(str(tmp_path / "tuned-2"), "int8", ["CPUExecutionProvider"])
    assert calls[-1][1]["quantization"] == "int8"
    asr._load("nemo-parakeet-tdt-0.6b-v3", None, ["CPUExecutionProvider"])
    assert calls[-1][0] == ("nemo-parakeet-tdt-0.6b-v3",)


def test_broken_tuned_model_falls_back_to_base(tmp_path, monkeypatch):
    d = make_model(tmp_path, "tuned-1")
    loaded = []

    def fake_load(model, quantization, providers):
        if model != asr.BASE_MODEL:
            raise RuntimeError("bad onnx file")
        loaded.append(model)
        return type("M", (), {"recognize": lambda self, *a, **k: ""})()

    monkeypatch.setattr(asr, "_load", fake_load)
    monkeypatch.setattr(asr, "select_providers", lambda device: (["CPUExecutionProvider"], "CPU"))
    t = asr.Transcriber(str(d), device="cpu")
    t.load()
    assert t.ready.is_set() and t.model_name == asr.BASE_MODEL and "bad onnx" in t.fallback_reason

    t2 = asr.Transcriber(str(tmp_path / "deleted-model"), device="cpu")  # folder removed
    t2.load()
    assert t2.model_name == asr.BASE_MODEL


def test_base_model_failure_still_raises(monkeypatch):
    monkeypatch.setattr(asr, "_load", lambda *a: (_ for _ in ()).throw(RuntimeError("no net")))
    monkeypatch.setattr(asr, "select_providers", lambda device: (["CPUExecutionProvider"], "CPU"))
    with pytest.raises(RuntimeError):
        asr.Transcriber(asr.BASE_MODEL, device="cpu").load()


# --- Settings section ---------------------------------------------------------------------------
class Ctl:
    def __init__(self, model=asr.BASE_MODEL):
        self.cfg = Config(model=model)
        self.updates, self.restarted = [], 0

    def update_setting(self, key, value):
        self.updates.append((key, value))
        setattr(self.cfg, key, value)

    def restart(self):
        self.restarted += 1


@pytest.fixture
def qapp():
    return QApplication.instance() or QApplication([])


def _texts(w):
    from PySide6.QtWidgets import QLabel

    return " | ".join(lab.text() for lab in w.findChildren(QLabel))


def test_installed_models_and_summary(tmp_path):
    make_model(tmp_path, "tuned-20261112-1030")
    make_model(tmp_path, "tuned-old", report=None)
    (tmp_path / "junk").mkdir()
    models = installed_models(tmp_path)
    assert [m.name for m in models] == ["tuned-old", "tuned-20261112-1030"]
    s = models[1].summary()
    assert "8.4% → 5.1%" in s and "6.0% → 6.1%" in s and "42 min" in s
    assert "Not evaluated" in models[0].summary()


def test_section_base_active_offers_tuned(tmp_path, qapp):
    d = make_model(tmp_path, "tuned-1")
    ctl = Ctl()
    sec = SpeechModelSection(ctl, root=tmp_path)
    assert "base model" in _texts(sec) and "Tuned model: tuned-1" in _texts(sec)
    sec.switch(str(d), "tuned-1", confirm=False)
    assert ctl.updates == [("model", str(d))] and ctl.restarted == 1


def test_section_tuned_active_switches_back(tmp_path, qapp):
    d = make_model(tmp_path, "tuned-1")
    ctl = Ctl(model=str(d))
    sec = SpeechModelSection(ctl, root=tmp_path)
    assert "Tuned to your voice: tuned-1" in _texts(sec) and "8.4% → 5.1%" in _texts(sec)
    sec.switch(asr.BASE_MODEL, "the base model", confirm=False)
    assert ctl.cfg.model == asr.BASE_MODEL and ctl.restarted == 1


def test_section_reports_fallback(tmp_path, qapp):
    ctl = Ctl(model=str(tmp_path / "gone"))
    ctl.transcriber = type("T", (), {"fallback_reason": "file not found"})()
    sec = SpeechModelSection(ctl, root=tmp_path)
    assert "base model is running" in _texts(sec)
