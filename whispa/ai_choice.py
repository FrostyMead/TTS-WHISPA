"""Pick the best local AI model for this computer, then check it's quick enough here.

1. Look at the hardware: Apple Silicon chip and memory, an NVIDIA GPU and its memory, or
   just the CPU and RAM.
2. Recommend a list of models, best first (see ``recommend``). Chosen from a comparison
   on Murmr's own prompts (clean-up, Command Mode, meeting summaries), Nov 2026:
   * gemma3:4b and qwen3.5:4b: both excellent. Qwen is Apache 2.0 licensed and has an
     MLX build that runs natively on Apple Silicon; Gemma was faster on an NVIDIA GPU.
   * qwen3.5:2b: safe (never mangles text) but leaves some self-corrections in. Best of
     the small ones: gemma3:1b and qwen3.5:0.8b invented words or obeyed dictated text.
3. After downloading, time a few real clean-ups (``speed_check``). Too slow for
   dictation: step down to the next smaller model.
"""

from __future__ import annotations

import logging
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

SLOW_S = 1.5  # a clean-up taking longer than this (median, warm) is too slow to wait for

# model -> (download GB, size class). A smaller class is what we step down to when slow.
MODELS = {
    "qwen3.5:4b-mlx": (4.0, 4), "qwen3.5:4b": (3.3, 4), "gemma3:4b": (3.3, 4),
    "qwen3.5:2b": (1.9, 2), "qwen3.5:2b-mlx": (3.1, 2),
}
KNOWN = set(MODELS) | {"gemma3:1b", "qwen3.5:0.8b"}  # models Murmr itself has installed

SPEED_SAMPLES = [
    "so um I think we should meet on tuesday no wait wednesday at 3",
    "can you send me the the report by friday",
    "the price is like twenty dollars I mean thirty dollars",
]


@dataclass
class Hardware:
    system: str = sys.platform
    arch: str = field(default_factory=platform.machine)
    ram_gb: float = 8.0
    chip: str = ""  # "Apple M2 Pro", "Intel(R) Core(TM) i7..."
    gpu: str = ""  # NVIDIA GPU name
    vram_gb: float = 0.0

    @property
    def apple_silicon(self) -> bool:
        return self.system == "darwin" and self.arch == "arm64"

    def describe(self) -> str:
        if self.apple_silicon:
            return f"{self.chip or 'Apple Silicon'}, {self.ram_gb:.0f} GB"
        if self.gpu:
            return f"{self.gpu}, {self.vram_gb:.0f} GB graphics memory"
        return f"{self.ram_gb:.0f} GB memory, no supported graphics card"


def _run(cmd: list[str]) -> str:
    flags = 0x08000000 if sys.platform == "win32" else 0  # CREATE_NO_WINDOW
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=5, creationflags=flags)
        return r.stdout.strip() if r.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def detect() -> Hardware:
    from .ai_setup import memory_gb

    hw = Hardware(ram_gb=round(memory_gb(), 1))
    if sys.platform == "darwin":
        hw.chip = _run(["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"])
    else:
        out = _run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"])
        if out:
            name, _, mem = out.splitlines()[0].rpartition(",")
            try:
                hw.gpu, hw.vram_gb = name.strip(), float(mem) / 1024
            except ValueError:
                pass
    return hw


@dataclass
class Choice:
    models: list[str]  # best first; later ones are fallbacks (errors) or smaller (too slow)
    reason: str

    @property
    def model(self) -> str:
        return self.models[0]


def recommend(hw: Hardware) -> Choice:
    if hw.apple_silicon:
        if hw.ram_gb >= 15:
            return Choice(["qwen3.5:4b-mlx", "qwen3.5:4b", "qwen3.5:2b"],
                          f"{hw.describe()}: the 4B model, built for Apple Silicon")
        return Choice(["qwen3.5:2b"], f"{hw.describe()}: the 2B model leaves room for dictation")
    if hw.system == "darwin":
        return Choice(["qwen3.5:2b"], f"{hw.describe()} (Intel Mac): the light 2B model")
    if hw.gpu and hw.vram_gb >= 6:
        return Choice(["gemma3:4b", "qwen3.5:4b", "qwen3.5:2b"],
                      f"{hw.describe()}: the 4B model on the graphics card")
    if hw.ram_gb >= 15:
        return Choice(["qwen3.5:2b"], f"{hw.describe()}: the 2B model runs well on the processor")
    return Choice(["qwen3.5:2b"], f"{hw.describe()}: the light 2B model")


def size_label(model: str) -> str:
    gb = MODELS.get(model, (0, 0))[0]
    return f"{gb:.1f} GB" if gb else "a few GB"


def smaller(models: list[str], current: str) -> str | None:
    """The first later model in the list that's in a smaller size class."""
    cls = MODELS.get(current, (0, 99))[1]
    after = models[models.index(current) + 1:] if current in models else models
    return next((m for m in after if MODELS.get(m, (0, 99))[1] < cls), None)


def speed_check(llm_cfg, model: str, cancel=None) -> float:
    """Median seconds for a real clean-up with ``model`` (after loading it)."""
    from dataclasses import replace

    from .llm import Polisher

    p = Polisher(replace(llm_cfg, model=model, timeout_s=60, keep_alive="2m"))
    p._complete("warm up", app="", terms=[])  # loads the model; not timed
    times = []
    for text in SPEED_SAMPLES:
        if cancel is not None and cancel.is_set():
            break
        t0 = time.perf_counter()
        p._complete(text, app="", terms=[])
        times.append(time.perf_counter() - t0)
    return statistics.median(times) if times else 0.0
