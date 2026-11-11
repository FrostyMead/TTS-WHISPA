"""Read text aloud with a natural local voice (Kokoro-82M), or the system voice.

* Kokoro: an 82M-parameter TTS model, downloaded on first use from Hugging Face. On an
  NVIDIA GPU: the full model (310 MB, ~15x faster than real time); otherwise the int8
  model (88 MB) on a few CPU threads. Pronunciation comes from espeak-ng.
  Sentences are generated one ahead of playback, so speech starts in about a second.
  The model is unloaded after a few idle minutes so it doesn't sit in memory.
* System voice: Windows SAPI, macOS ``say``, Linux ``spd-say``/``espeak-ng``. Instant,
  no download; used when Kokoro isn't available.
"""

from __future__ import annotations

import ctypes
import json
import logging
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
from typing import Callable

import numpy as np

log = logging.getLogger(__name__)

REPO = "onnx-community/Kokoro-82M-v1.0-ONNX"
MODEL_GPU = "onnx/model.onnx"  # fp16 gives silence (NaN) on CUDA
MODEL_CPU = "onnx/model_quantized.onnx"
SAMPLE_RATE = 24000
IDLE_UNLOAD_S = 180
MAX_TOKENS = 400  # the model takes up to 510; shorter pieces start playing sooner

# (voice id, label). a/b = American/British, f/m = female/male.
VOICES = [
    ("af_heart", "Heart · American"), ("af_bella", "Bella · American"),
    ("af_nicole", "Nicole · American, soft"), ("am_michael", "Michael · American"),
    ("am_fenrir", "Fenrir · American"), ("bf_emma", "Emma · British"),
    ("bf_isabella", "Isabella · British"), ("bm_george", "George · British"),
    ("bm_fable", "Fable · British"), ("system", "System voice · no download"),
]


# --------------------------------------------------------------------------- espeak-ng
class Espeak:
    """Text -> IPA phonemes through the espeak-ng library (ctypes, no extra packages)."""

    _PUNCT = re.compile(r'([.,!?;:—…"()“”]+)')

    def __init__(self, voice: str = "en-us"):
        import espeakng_loader

        lib_path = espeakng_loader.get_library_path()
        data = espeakng_loader.get_data_path()
        self.lib = ctypes.CDLL(lib_path)
        self.lib.espeak_Initialize.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
        self.lib.espeak_SetVoiceByName.argtypes = [ctypes.c_char_p]
        self.lib.espeak_TextToPhonemes.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int,
                                                   ctypes.c_int]
        self.lib.espeak_TextToPhonemes.restype = ctypes.c_char_p
        # The path is the folder that contains espeak-ng-data. 0x8000: don't exit on errors.
        root = os.path.dirname(data).encode()
        if self.lib.espeak_Initialize(2, 0, root, 0x8000) < 0:  # AUDIO_OUTPUT_SYNCHRONOUS
            raise RuntimeError("espeak-ng failed to start")
        self.voice = ""
        self.set_voice(voice)

    def set_voice(self, voice: str) -> None:
        if voice != self.voice and self.lib.espeak_SetVoiceByName(voice.encode()) == 0:
            self.voice = voice

    def _clause_phonemes(self, text: str) -> str:
        buf = ctypes.create_string_buffer(text.encode("utf-8"))
        ptr = ctypes.c_void_p(ctypes.addressof(buf))
        out = []
        while ptr.value:
            res = self.lib.espeak_TextToPhonemes(ctypes.byref(ptr), 1, 0x02)  # UTF-8 in, IPA out
            if res:
                out.append(res.decode("utf-8", "replace"))
        return " ".join(out)

    def phonemize(self, text: str) -> str:
        """IPA with stress marks; punctuation is kept (espeak drops it, Kokoro uses it)."""
        parts = []
        for piece in self._PUNCT.split(text):
            if not piece.strip():
                continue
            if self._PUNCT.fullmatch(piece):
                parts.append(piece.strip())
                parts.append(" ")
            else:
                parts.append(self._clause_phonemes(piece.strip()))
        ps = "".join(parts).strip()
        # Kokoro's own clean-ups for English espeak output.
        ps = ps.replace("ʲ", "j").replace("r", "ɹ").replace("x", "k").replace("ɬ", "l")
        ps = re.sub(r"(?<=[a-zɹː])(?=hˈʌndɹɪd)", " ", ps)
        ps = re.sub(r' z(?=[;:,.!?¡¿—…"«»“” ]|$)', "z", ps)
        return re.sub(r"\s+", " ", ps)


# --------------------------------------------------------------------------- Kokoro
def split_sentences(text: str) -> list[str]:
    text = re.sub(r"https?://\S+", "link", text)  # don't spell out URLs
    out = []
    for para in re.split(r"\n\s*\n|\n", text):
        para = para.strip(" \t-•*#>")
        if para:
            out += [s.strip() for s in re.split(r"(?<=[.!?…])\s+", para) if s.strip()]
    return out


class Kokoro:
    def __init__(self, device: str = "auto", threads: int | None = None):
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download

        from .asr import Transcriber, select_providers

        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        self._download = lambda f: hf_hub_download(REPO, f)
        with open(self._download("tokenizer.json"), encoding="utf-8") as f:
            self.vocab: dict[str, int] = json.load(f)["model"]["vocab"]
        providers, label = select_providers(device)
        gpu = label == "CUDA"  # DirectML/CoreML: CPU is the safer choice for this model
        so, tuned = Transcriber._lean_options(providers if gpu else ["CPUExecutionProvider"])
        # Light on the machine: a few threads.
        so.intra_op_num_threads = threads or max(1, min(4, (os.cpu_count() or 4) // 2))
        so.inter_op_num_threads = 1
        self.backend = label if gpu else "CPU"
        self.session = ort.InferenceSession(self._download(MODEL_GPU if gpu else MODEL_CPU), so,
                                            providers=tuned)
        self.espeak = Espeak()
        self._voices: dict[str, np.ndarray] = {}
        self.synth(self.tokens("Hi."), "af_heart", 1.0)  # warm-up: the first run is slow
        log.info("Reading voice ready on %s", self.backend)

    def voice(self, name: str) -> np.ndarray:
        if name not in self._voices:
            path = self._download(f"voices/{name}.bin")
            self._voices[name] = np.fromfile(path, dtype=np.float32).reshape(-1, 1, 256)
        return self._voices[name]

    def tokens(self, text: str) -> list[int]:
        return [self.vocab[c] for c in self.espeak.phonemize(text) if c in self.vocab]

    def pieces(self, text: str) -> list[list[int]]:
        """Token lists of at most MAX_TOKENS, split at sentence ends (or words)."""
        out: list[list[int]] = []
        for sentence in split_sentences(text):
            toks = self.tokens(sentence)
            while len(toks) > MAX_TOKENS:  # very long sentence: cut at a space
                cut = max((i for i, t in enumerate(toks[:MAX_TOKENS]) if t == self.vocab.get(" ")),
                          default=MAX_TOKENS)
                out.append(toks[:cut])
                toks = toks[cut + 1:]
            if toks:
                if out and len(out[-1]) + len(toks) < 120:  # join short sentences
                    out[-1] = out[-1] + [self.vocab.get(" ", 16)] + toks
                else:
                    out.append(toks)
        return out

    def synth(self, toks: list[int], voice: str, speed: float) -> np.ndarray:
        style = self.voice(voice)[min(len(toks), 509)]
        names = {i.name for i in self.session.get_inputs()}
        feeds = {"input_ids" if "input_ids" in names else "tokens": np.array([[0, *toks, 0]], np.int64),
                 "style": style.astype(np.float32),
                 "speed": np.array([speed], np.float32)}
        return self.session.run(None, feeds)[0].reshape(-1).astype(np.float32)


# --------------------------------------------------------------------------- speaker
class Speaker:
    """Reads text aloud in the background. ``speak()`` again or ``stop()`` interrupts."""

    def __init__(self, voice: str = "af_heart", speed: float = 1.0, device: str = "auto",
                 on_state: Callable[[str], None] | None = None):
        self.voice, self.speed, self.device = voice, speed, device
        self.on_state = on_state or (lambda s: None)  # "loading" | "speaking" | "idle" | "error:..."
        self._kokoro: Kokoro | None = None
        self._kokoro_failed = False
        self._lock = threading.Lock()
        self._gen = 0  # bumps on every speak/stop: older playback threads quit
        self._system_proc = None
        self._last_used = 0.0
        self.speaking = False
        threading.Thread(target=self._unload_when_idle, name="tts-idle", daemon=True).start()

    # -- public ----------------------------------------------------------
    def speak(self, text: str) -> None:
        self.stop()
        text = text.strip()
        if not text:
            return
        with self._lock:
            self._gen += 1
            gen = self._gen
        self.speaking = True
        threading.Thread(target=self._run, args=(text, gen), name="tts", daemon=True).start()

    def stop(self) -> None:
        with self._lock:
            self._gen += 1
        self._stop_system()
        if self.speaking:
            self.speaking = False
            self.on_state("idle")

    def unload(self) -> None:
        with self._lock:
            self._kokoro = None

    # -- internals -------------------------------------------------------
    def _current(self, gen: int) -> bool:
        return gen == self._gen

    def _run(self, text: str, gen: int) -> None:
        try:
            if self.voice != "system" and not self._kokoro_failed:
                try:
                    self._run_kokoro(text, gen)
                    return
                except ImportError as exc:
                    log.warning("Natural voice unavailable (%s); using the system voice", exc)
                    self._kokoro_failed = True
                except Exception as exc:  # noqa: BLE001 - offline download, bad model...
                    log.exception("Natural voice failed; using the system voice")
                    if self._kokoro is None:
                        self._kokoro_failed = True
                    if not self._current(gen):
                        return
                    self.on_state(f"error:Natural voice failed ({exc}); using the system voice")
            if self._current(gen):
                self._run_system(text, gen)
        finally:
            if self._current(gen):
                self.speaking = False
                self.on_state("idle")

    def _engine(self) -> Kokoro:
        with self._lock:
            k = self._kokoro
        if k is None:
            self.on_state("loading")
            k = Kokoro(self.device)
            with self._lock:
                self._kokoro = k
        return k

    def _run_kokoro(self, text: str, gen: int) -> None:
        import sounddevice as sd

        k = self._engine()
        pieces = k.pieces(text)
        audio_q: queue.Queue = queue.Queue(maxsize=2)  # generate one piece ahead, no more

        def produce():
            try:
                for toks in pieces:
                    if not self._current(gen):
                        break
                    audio_q.put(k.synth(toks, self.voice, self.speed))
            except Exception as exc:  # noqa: BLE001
                audio_q.put(exc)
            audio_q.put(None)

        threading.Thread(target=produce, name="tts-gen", daemon=True).start()
        self.on_state("speaking")
        with sd.OutputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32") as out:
            while True:
                item = audio_q.get()
                if item is None or not self._current(gen):
                    break
                if isinstance(item, Exception):
                    raise item
                for i in range(0, len(item), SAMPLE_RATE // 10):  # 100 ms: stop is quick
                    if not self._current(gen):
                        out.abort()
                        return
                    out.write(item[i:i + SAMPLE_RATE // 10])
                self._last_used = time.monotonic()
        self._last_used = time.monotonic()

    def _run_system(self, text: str, gen: int) -> None:
        self.on_state("speaking")
        rate = self.speed
        if sys.platform == "darwin":
            cmd = ["say", "-r", str(int(185 * rate)), text]
        elif sys.platform == "win32":
            script = ("Add-Type -AssemblyName System.Speech;"
                      "$s=New-Object System.Speech.Synthesis.SpeechSynthesizer;"
                      f"$s.Rate={max(-10, min(10, round((rate - 1) * 10)))};"
                      "$s.Speak([Console]::In.ReadToEnd())")
            cmd = ["powershell", "-NoProfile", "-NonInteractive", "-Command", script]
        else:
            tool = shutil.which("spd-say") or shutil.which("espeak-ng") or shutil.which("espeak")
            if not tool:
                raise RuntimeError("No system voice found (install speech-dispatcher or espeak-ng)")
            cmd = [tool, "-w", text] if tool.endswith("spd-say") else [tool, text]
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL, creationflags=flags)
        self._system_proc = proc
        if sys.platform == "win32":
            proc.stdin.write(text.encode("utf-8", "replace"))
        proc.stdin.close()
        while proc.poll() is None and self._current(gen):
            time.sleep(0.1)
        self._stop_system()

    def _stop_system(self) -> None:
        proc, self._system_proc = self._system_proc, None
        if proc is not None and proc.poll() is None:
            proc.kill()

    def _unload_when_idle(self) -> None:
        while True:
            time.sleep(30)
            if (self._kokoro is not None and not self.speaking
                    and time.monotonic() - self._last_used > IDLE_UNLOAD_S):
                log.info("Unloading the reading voice (idle)")
                self.unload()
