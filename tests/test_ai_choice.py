"""Picking the local AI model for the hardware, and stepping down when it's too slow."""
import threading

import pytest

from whispa import ai_choice, ai_setup
from whispa.ai_choice import Hardware, recommend, smaller
from whispa.config import LLMCleanup


@pytest.mark.parametrize("hw,first", [
    (Hardware(system="darwin", arch="arm64", ram_gb=16, chip="Apple M2"), "qwen3.5:4b-mlx"),
    (Hardware(system="darwin", arch="arm64", ram_gb=8, chip="Apple M1"), "qwen3.5:2b"),
    (Hardware(system="darwin", arch="x86_64", ram_gb=16), "qwen3.5:2b"),
    (Hardware(system="win32", ram_gb=32, gpu="NVIDIA GeForce RTX 3070 Ti", vram_gb=8), "gemma3:4b"),
    (Hardware(system="win32", ram_gb=32, gpu="NVIDIA GeForce GTX 1050", vram_gb=2), "qwen3.5:2b"),
    (Hardware(system="win32", ram_gb=8), "qwen3.5:2b"),
])
def test_recommendation_fits_the_hardware(hw, first):
    choice = recommend(hw)
    assert choice.model == first and choice.reason


def test_smaller_skips_same_size_alternatives():
    models = ["qwen3.5:4b-mlx", "qwen3.5:4b", "qwen3.5:2b"]
    assert smaller(models, "qwen3.5:4b-mlx") == "qwen3.5:2b"
    assert smaller(models, "qwen3.5:2b") is None


@pytest.fixture
def fake_ollama(monkeypatch):
    state = {"installed": {"gemma3:4b"}, "pulled": [], "removed": [], "speed": {}, "broken": set()}
    monkeypatch.setattr("whispa.llm.find_ollama", lambda: "ollama")
    monkeypatch.setattr(ai_setup, "ensure_server", lambda *a: None)
    monkeypatch.setattr(ai_choice, "detect",
                        lambda: Hardware(system="darwin", arch="arm64", ram_gb=16, chip="Apple M2"))

    def pull(base, model, progress, cancel):
        if model in state["broken"]:
            raise RuntimeError("this runner isn't supported")
        state["pulled"].append(model)
        state["installed"].add(model)

    def remove(base, model):
        state["removed"].append(model)
        state["installed"].discard(model)

    monkeypatch.setattr(ai_setup, "pull", pull)
    monkeypatch.setattr(ai_setup, "remove_model", remove)
    monkeypatch.setattr(ai_setup, "installed_models", lambda base: set(state["installed"]))
    monkeypatch.setattr(ai_choice, "speed_check", lambda cfg, model, cancel=None: state["speed"].get(model, 0.3))
    return state


def test_mac_switches_to_the_best_model_and_frees_the_old_one(fake_ollama):
    cfg = LLMCleanup(model="gemma3:4b")
    assert ai_setup.setup(cfg, lambda *a: None, threading.Event()) == "qwen3.5:4b-mlx"
    assert cfg.measured_s == 0.3
    assert fake_ollama["removed"] == ["gemma3:4b"]


def test_broken_mlx_build_falls_back_to_the_regular_one(fake_ollama):
    fake_ollama["broken"].add("qwen3.5:4b-mlx")
    assert ai_setup.setup(LLMCleanup(model=""), lambda *a: None, threading.Event()) == "qwen3.5:4b"


def test_too_slow_steps_down_and_removes_what_it_downloaded(fake_ollama):
    fake_ollama["speed"]["qwen3.5:4b-mlx"] = 4.0
    cfg = LLMCleanup(model="gemma3:4b")
    msgs = []
    assert ai_setup.setup(cfg, lambda m, f: msgs.append(m), threading.Event()) == "qwen3.5:2b"
    assert any("lighter" in m for m in msgs)
    assert set(fake_ollama["removed"]) == {"gemma3:4b", "qwen3.5:4b-mlx"}


def test_a_model_you_chose_is_left_alone(fake_ollama):
    cfg = LLMCleanup(model="llama3.2:3b", auto_model=False)
    assert ai_setup.setup(cfg, lambda *a: None, threading.Event()) == "llama3.2:3b"
    assert fake_ollama["removed"] == []
