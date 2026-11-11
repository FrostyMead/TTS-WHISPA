import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from whispa.config import LLMCleanup
from whispa.llm import Polisher


@pytest.fixture
def fake_ollama():
    """A stand-in for Ollama's /api/tags and /api/chat."""
    state = {"reply": "So let's meet on Wednesday.", "requests": []}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, obj):
            body = json.dumps(obj).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self._send({"models": [{"name": "gemma3:4b"}]})

        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state["requests"].append(req)
            self._send({"message": {"role": "assistant", "content": state["reply"]}})

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    state["url"] = f"http://127.0.0.1:{server.server_port}"
    yield state
    server.shutdown()


def test_polish_uses_ollama(fake_ollama):
    p = Polisher(LLMCleanup(base_url=fake_ollama["url"]))
    p.refresh()  # availability is normally checked in the background
    out = p.polish("so let's meet on tuesday no wait wednesday", app="Slack", terms=["Kubernetes"])
    assert out == "So let's meet on Wednesday."
    req = fake_ollama["requests"][-1]
    assert req["model"] == "gemma3:4b" and req["stream"] is False and req["think"] is False
    system = req["messages"][0]["content"]
    assert "Slack" in system and "Kubernetes" in system
    assert "tuesday no wait wednesday" in req["messages"][-1]["content"]


def test_rejects_answers_instead_of_edits(fake_ollama):
    fake_ollama["reply"] = "The capital of France is Paris. " * 10
    p = Polisher(LLMCleanup(base_url=fake_ollama["url"]))
    p.refresh()
    assert p.polish("what's the capital of france") == "what's the capital of france"


def test_auto_off_when_no_server():
    p = Polisher(LLMCleanup(base_url="http://127.0.0.1:9"))
    assert p.polish("hello there") == "hello there"  # never blocks on the check
    assert p.active(wait=True) is False


def test_check_runs_in_background(fake_ollama):
    import time

    p = Polisher(LLMCleanup(base_url=fake_ollama["url"]))
    assert p.active() is False  # unknown yet: don't wait
    for _ in range(50):
        if p.available is not None:
            break
        time.sleep(0.02)
    assert p.active() is True


def test_auto_off_when_model_not_pulled(fake_ollama):
    p = Polisher(LLMCleanup(base_url=fake_ollama["url"], model="qwen3:4b"))
    assert not p.active(wait=True)
