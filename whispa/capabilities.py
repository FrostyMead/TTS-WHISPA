"""What works on this computer, so Murmr only shows features that do.

* Tuning the speech model and training a voice need an NVIDIA GPU with Linux / WSL2, so
  not macOS. (Choosing a GPU backend doesn't apply there either.)
* Command Mode, AI clean-up and meeting summaries need the local AI (Ollama, or a server
  you configured). Installed later, they appear by themselves.
"""

from __future__ import annotations

import sys
import time

_ai_cache: tuple[float, bool] = (0.0, False)


def can_train() -> bool:
    return sys.platform != "darwin"


def gpu_choice() -> bool:
    return sys.platform != "darwin"


def local_ai_installed(llm_cfg) -> bool:
    """Ollama is installed (or you pointed Murmr at another AI server). Checked at most every
    30 s: it's a few file lookups."""
    global _ai_cache
    if llm_cfg.provider != "ollama" or not any(h in llm_cfg.base_url for h in ("127.0.0.1", "localhost")):
        return True  # a server you configured yourself
    now = time.monotonic()
    if now - _ai_cache[0] > 30:
        from .llm import find_ollama

        _ai_cache = (now, find_ollama() is not None)
    return _ai_cache[1]
