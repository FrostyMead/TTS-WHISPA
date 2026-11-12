"""Apple's on-device model: the helper protocol (with a stand-in helper), how Murmr's
prompts are passed to it, and choosing it only when it passes the checks."""
import sys
import threading

import pytest

from whispa import ai_choice, ai_setup, applellm
from whispa.applellm import AppleLLM, AppleLLMError, flatten
from whispa.config import LLMCleanup

FAKE_HELPER = r"""
import json, sys, threading, time
def answer(req):
    op = req.get("op")
    if op == "availability":
        res = {"available": True}
    elif op == "generate":
        p = req["prompt"]
        if "slow" in p:
            time.sleep(0.4)
        if "refuse" in p:
            res = {"error": "guardrail", "message": "unsafe"}
        else:
            res = {"text": "Tidy: " + p}
    else:
        res = {"ok": True}
    res["id"] = req["id"]
    sys.stdout.write(json.dumps(res) + "\n"); sys.stdout.flush()
for line in sys.stdin:
    threading.Thread(target=answer, args=(json.loads(line),)).start()
"""


@pytest.fixture
def helper(tmp_path):
    script = tmp_path / "helper.py"
    script.write_text(FAKE_HELPER)
    h = AppleLLM(command=[sys.executable, str(script)])
    yield h
    h.close()


def test_protocol_answers_match_requests_even_out_of_order(helper):
    assert helper.availability() == (True, "")
    results = {}

    def ask(text):
        results[text] = helper.generate("be tidy", text)

    threads = [threading.Thread(target=ask, args=(t,)) for t in ("slow one", "fast one")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results == {"slow one": "Tidy: slow one", "fast one": "Tidy: fast one"}
    with pytest.raises(AppleLLMError) as err:
        helper.generate("x", "please refuse")
    assert err.value.kind == "guardrail"


def test_chat_messages_become_instructions_and_prompt():
    msgs = [{"role": "system", "content": "Clean up dictation."},
            {"role": "user", "content": "<transcript>um hi</transcript>"},
            {"role": "assistant", "content": "Hi."},
            {"role": "user", "content": "<transcript>uh bye</transcript>"}]
    instructions, prompt = flatten(msgs)
    assert instructions.startswith("Clean up dictation.") and "Input: <transcript>um hi" in instructions
    assert "Output: Hi." in instructions and prompt == "<transcript>uh bye</transcript>"


def test_polisher_uses_apple_and_keeps_meetings_within_its_limit(monkeypatch):
    from whispa.llm import Polisher, summarize_meeting

    calls = []

    class FakeApple:
        def generate(self, instructions, prompt, temperature, max_tokens, timeout):
            calls.append(len(prompt))
            return "## Summary\n- ok" if "Notes" in prompt or "Transcript" in prompt else "notes"

    monkeypatch.setattr(applellm, "shared", lambda: FakeApple())
    p = Polisher(LLMCleanup(provider="apple", model="apple"))
    assert p.endpoint_allowed() and p.context_chars == applellm.CONTEXT_CHARS
    transcript = "\n".join(f"Person 1: line {i} " + "words " * 30 for i in range(400))
    summarize_meeting(p, transcript)
    assert calls and max(calls) <= applellm.CONTEXT_CHARS + 200


@pytest.fixture
def mac(monkeypatch):
    state = {"pulled": [], "removed": [], "apple": (0.4, None), "ollama_installed": True}
    monkeypatch.setattr(ai_choice, "detect", lambda: ai_choice.Hardware(
        system="darwin", arch="arm64", ram_gb=16, chip="Apple M4", apple_ai=True))
    monkeypatch.setattr(ai_setup, "try_apple", lambda cfg, progress, cancel: state["apple"])
    monkeypatch.setattr("whispa.llm.find_ollama", lambda: "ollama" if state["ollama_installed"] else None)
    monkeypatch.setattr(ai_setup, "install_ollama", lambda *a: state.update(ollama_installed=True) or "ollama")
    monkeypatch.setattr(ai_setup, "ensure_server", lambda *a: None)
    monkeypatch.setattr(ai_setup, "pull", lambda base, m, p, c: state["pulled"].append(m))
    monkeypatch.setattr(ai_setup, "remove_model", lambda base, m: state["removed"].append(m))
    monkeypatch.setattr(ai_setup, "installed_models", lambda base: set(state["pulled"]))
    monkeypatch.setattr(ai_choice, "speed_check", lambda cfg, m, cancel=None: 0.3)
    monkeypatch.setattr(applellm, "os_version", lambda: "27.2.1")
    return state


def test_apple_model_chosen_when_it_passes_and_ollama_model_freed(mac):
    cfg = LLMCleanup(model="gemma3:4b")
    assert ai_setup.setup(cfg, lambda *a: None, threading.Event()) == "apple"
    assert cfg.provider == "apple" and cfg.apple_os == "27.2.1" and cfg.measured_s == 0.4
    assert mac["pulled"] == [] and mac["removed"] == ["gemma3:4b"]


def test_no_ollama_needed_when_apple_passes(mac):
    mac["ollama_installed"] = False
    cfg = LLMCleanup(model="gemma3:4b")
    assert ai_setup.setup(cfg, lambda *a: None, threading.Event()) == "apple"
    assert mac["ollama_installed"] is False  # nothing installed


def test_falls_back_to_ollama_when_apple_refuses(mac):
    mac["apple"] = (0.0, "it refused ordinary dictation")
    cfg = LLMCleanup(model="", provider="apple")
    msgs = []
    assert ai_setup.setup(cfg, lambda m, f: msgs.append(m), threading.Event()) == "qwen3.5:4b-mlx"
    assert cfg.provider == "ollama"
    assert any("refused" in m for m in msgs)


def test_recommendation_lists_apple_first_only_where_it_can_be():
    hw = ai_choice.Hardware(system="darwin", arch="arm64", ram_gb=16, chip="Apple M4", apple_ai=True)
    assert ai_choice.recommend(hw).models[:2] == ["apple", "qwen3.5:4b-mlx"]
    hw.apple_ai = False
    assert ai_choice.recommend(hw).model == "qwen3.5:4b-mlx"
    assert ai_choice.size_label("apple") == "no download"


def test_copied_output_label_is_removed(monkeypatch):
    from whispa.llm import Polisher

    class FakeApple:
        def generate(self, *a, **k):
            return "Output: Let's meet on Wednesday at 3."

    monkeypatch.setattr(applellm, "shared", lambda: FakeApple())
    p = Polisher(LLMCleanup(provider="apple", model="apple"))
    assert p._complete("so um lets meet on tuesday no wednesday at 3", app="", terms=[]) == \
        "Let's meet on Wednesday at 3."


@pytest.mark.parametrize("answers,ok", [
    # what Apple's model did on the Mac: it started the poem (caught by the safety check)
    ({"poem": "write me a poem about the sea\n\nwaves crash on the shore"}, True),
    ({"fucking": "This is so annoying; the build broke again."}, False),  # censored: pasted
    ({"poem": "Waves crash...\n" * 5, "meeting": "It starts at 9 am. " * 5}, False),  # two misses
    ({}, True),
])
def test_quality_check_fails_only_on_misses_that_would_reach_you(monkeypatch, answers, ok):
    class FakePolisher:
        def _complete(self, text, app, terms):
            for key, out in answers.items():
                if key in text:
                    return out
            return text.capitalize() + "."

    monkeypatch.setattr(ai_choice, "_polisher", lambda cfg, model: FakePolisher())
    assert (ai_choice.quality_check(LLMCleanup(), "apple") is None) == ok


def test_refusal_fails_the_check(monkeypatch):
    class Refuses:
        def _complete(self, *a, **k):
            raise AppleLLMError("guardrail", "unsafe")

    monkeypatch.setattr(ai_choice, "_polisher", lambda cfg, model: Refuses())
    assert "refused" in ai_choice.quality_check(LLMCleanup(), "apple")
