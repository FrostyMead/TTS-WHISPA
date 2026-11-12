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
from pathlib import Path
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
MY_VOICE = "my:"  # voice ids for voices built from your recordings: "my:<folder name>"


def my_voices() -> list[tuple[str, str]]:
    """Voices built from your own Voice training recordings (<config>/voices/<name>/voice.onnx)."""
    from . import config as config_mod

    root = config_mod.config_dir() / "voices"
    out = []
    for d in sorted(root.iterdir()) if root.exists() else []:
        if (d / "voice.onnx").is_file() and (d / "voice.onnx.json").is_file():
            label = "My voice"
            try:
                info = json.loads((d / "info.json").read_text(encoding="utf-8"))
                if info.get("label"):
                    label = info["label"]
            except (OSError, ValueError):
                pass
            out.append((MY_VOICE + d.name, f"{label} · from your recordings"))
    return out


def all_voices() -> list[tuple[str, str]]:
    return my_voices() + VOICES


class PiperVoice:
    """A Piper (VITS) voice: the kind trained on your recordings. Small and fast on any CPU."""

    PAD, BOS, EOS = "_", "^", "$"

    def __init__(self, folder: Path, threads: int | None = None):
        import onnxruntime as ort

        self.folder = Path(folder)
        self.config = json.loads((self.folder / "voice.onnx.json").read_text(encoding="utf-8"))
        self.sample_rate = int(self.config["audio"]["sample_rate"])
        self.id_map: dict[str, list[int]] = self.config["phoneme_id_map"]
        inf = self.config.get("inference", {})
        self.noise_scale = float(inf.get("noise_scale", 0.667))
        self.length_scale = float(inf.get("length_scale", 1.0))
        self.noise_w = float(inf.get("noise_w", 0.8))
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads or max(1, min(4, (os.cpu_count() or 4) // 2))
        so.inter_op_num_threads = 1
        so.enable_cpu_mem_arena = False
        # Don't keep threads spinning on the CPU after each run, waiting for the next one: an
        # app that's mostly idle would burn battery for a few milliseconds of latency.
        so.add_session_config_entry("session.intra_op.allow_spinning", "0")
        so.add_session_config_entry("session.inter_op.allow_spinning", "0")
        self.session = ort.InferenceSession(str(self.folder / "voice.onnx"), so,
                                            providers=["CPUExecutionProvider"])
        self.espeak = Espeak(self.config.get("espeak", {}).get("voice", "en-us"))

    def ids(self, text: str) -> list[int]:
        import unicodedata

        phonemes = unicodedata.normalize("NFD", self.espeak.phonemize(text, kokoro=False))
        m = self.id_map
        out = [*m[self.BOS], *m[self.PAD]]
        for ph in phonemes:
            if ph in m:
                out += [*m[ph], *m[self.PAD]]
        return out + [*m[self.EOS]]

    def synth(self, text: str, speed: float = 1.0) -> np.ndarray:
        ids = self.ids(text)
        feeds = {"input": np.array([ids], np.int64),
                 "input_lengths": np.array([len(ids)], np.int64),
                 "scales": np.array([self.noise_scale, self.length_scale / max(0.5, speed),
                                     self.noise_w], np.float32)}
        if "sid" in {i.name for i in self.session.get_inputs()}:
            feeds["sid"] = np.array([0], np.int64)
        return self.session.run(None, feeds)[0].reshape(-1).astype(np.float32)


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

    def phonemize(self, text: str, kokoro: bool = True) -> str:
        """IPA with stress marks; punctuation is kept (espeak drops it, the voices use it).
        ``kokoro=False``: espeak's IPA as is (Piper voices are trained on exactly that)."""
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
        if not kokoro:
            return re.sub(r"\s+", " ", ps)
        # Kokoro's own clean-ups for English espeak output.
        ps = ps.replace("ʲ", "j").replace("r", "ɹ").replace("x", "k").replace("ɬ", "l")
        ps = re.sub(r"(?<=[a-zɹː])(?=hˈʌndɹɪd)", " ", ps)
        ps = re.sub(r' z(?=[;:,.!?¡¿—…"«»“” ]|$)', "z", ps)
        return re.sub(r"\s+", " ", ps)


# --------------------------------------------------------------------------- Kokoro
def mostly_cjk(text: str) -> str | None:
    """"zh" / "ja" / "ko" when most of the text is Chinese, Japanese or Korean (the natural
    English voices can't read those), else None."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return None
    han = sum(0x3400 <= ord(c) <= 0x9FFF or 0xF900 <= ord(c) <= 0xFAFF for c in letters)
    kana = sum(0x3040 <= ord(c) <= 0x30FF for c in letters)
    hangul = sum(0xAC00 <= ord(c) <= 0xD7AF or 0x1100 <= ord(c) <= 0x11FF for c in letters)
    if (han + kana + hangul) / len(letters) < 0.3:
        return None
    return "ko" if hangul > han + kana else "ja" if kana else "zh"


_MAC_VOICES: dict[str, str | None] = {}


def mac_voice_for(lang: str) -> str | None:
    """An installed macOS voice for ``lang`` ("zh" prefers mainland, then Taiwan, Hong Kong)."""
    if lang in _MAC_VOICES:
        return _MAC_VOICES[lang]
    prefs = {"zh": ("zh_CN", "zh_TW", "zh_HK"), "ja": ("ja_JP",), "ko": ("ko_KR",)}.get(lang, ())
    found: dict[str, str] = {}
    try:
        out = subprocess.run(["say", "-v", "?"], capture_output=True, text=True, timeout=10).stdout
        for line in out.splitlines():
            m = re.match(r"^(.+?)\s+([a-z]{2}_[A-Z]{2})\s+#", line)
            if m:
                found.setdefault(m.group(2), m.group(1).strip())
    except (OSError, subprocess.SubprocessError):
        pass
    voice = next((found[p] for p in prefs if p in found), None)
    _MAC_VOICES[lang] = voice
    return voice


LANG_NAMES = {"zh": "Chinese", "ja": "Japanese", "ko": "Korean"}
_WIN_VOICES: dict[str, bool] = {}


def _win_has_voice(lang: str) -> bool:
    """Is a Windows (SAPI) voice for ``lang`` installed? Asked once."""
    if lang not in _WIN_VOICES:
        script = ("Add-Type -AssemblyName System.Speech;"
                  "(New-Object System.Speech.Synthesis.SpeechSynthesizer).GetInstalledVoices()|"
                  "%{$_.VoiceInfo.Culture.Name}")
        try:
            out = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
                                 capture_output=True, text=True, timeout=20,
                                 creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout
            _WIN_VOICES[lang] = any(c.strip().lower().startswith(lang) for c in out.splitlines())
        except (OSError, subprocess.SubprocessError):
            _WIN_VOICES[lang] = True  # can't tell: don't nag
    return _WIN_VOICES[lang]


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"https?://\S+", "link", text)  # don't spell out URLs
    out = []
    for para in re.split(r"\n\s*\n|\n", text):
        para = para.strip(" \t-•*#>")
        if para:
            out += [s.strip() for s in re.split(r"(?<=[.!?…])\s+", para) if s.strip()]
    return out


def model_file(device: str = "auto") -> str:
    from .asr import select_providers

    return MODEL_GPU if select_providers(device)[1] == "CUDA" else MODEL_CPU


def voice_downloaded(voice: str, device: str = "auto") -> bool:
    """Are the model and this voice already on disk (so loading takes seconds, not a download)?"""
    try:
        from huggingface_hub import try_to_load_from_cache

        return all(isinstance(try_to_load_from_cache(REPO, f), str)
                   for f in (model_file(device), "tokenizer.json", f"voices/{voice}.bin"))
    except Exception:  # noqa: BLE001
        return False


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
        self._piper: PiperVoice | None = None
        self._kokoro_failed = False
        self._preparing = False
        self._lock = threading.Lock()
        self._gen = 0  # bumps on every speak/stop: older playback threads quit
        self._system_proc = None
        self._last_used = 0.0
        self.speaking = False
        self.paused = False
        self.level = 0.0  # loudness of what's playing, for the pill
        self.engine = ""  # "natural" | "system" while speaking
        self._skip = False
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
        self.speaking, self.paused, self._skip = True, False, False
        threading.Thread(target=self._run, args=(text, gen), name="tts", daemon=True).start()

    def stop(self) -> None:
        with self._lock:
            self._gen += 1
        self._stop_system()
        self.paused, self.level = False, 0.0
        if self.speaking:
            self.speaking = False
            self.on_state("idle")

    def toggle_pause(self) -> None:
        if not self.speaking:
            return
        self.paused = not self.paused
        proc = self._system_proc
        if proc is not None and proc.poll() is None:
            _suspend(proc, self.paused)
        self.on_state("paused" if self.paused else "resumed")

    @property
    def can_skip(self) -> bool:
        return self.engine == "natural"

    def skip(self) -> None:
        """Jump to the next sentence (natural voice)."""
        if self.speaking and self.can_skip:
            self._skip = True
            if self.paused:
                self.toggle_pause()

    def unload(self) -> None:
        with self._lock:
            self._kokoro = None
            self._piper = None

    # -- internals -------------------------------------------------------
    def _current(self, gen: int) -> bool:
        return gen == self._gen

    def prepare(self) -> None:
        """Download and load the natural voice in the background (first use)."""
        with self._lock:
            if self._preparing or self._kokoro is not None or self._kokoro_failed:
                return
            self._preparing = True

        def work():
            try:
                log.info("Read aloud: downloading/loading the natural voice in the background")
                k = Kokoro(self.device)
                k.voice(self.voice)
                with self._lock:
                    self._kokoro = k
                self._last_used = time.monotonic()
                self.on_state("ready")
            except Exception as exc:  # noqa: BLE001
                log.exception("Read aloud: natural voice unavailable; the system voice is used")
                self._kokoro_failed = True
                self.on_state(f"error:Natural voice unavailable ({exc}); using the system voice")
            finally:
                self._preparing = False

        threading.Thread(target=work, name="tts-prepare", daemon=True).start()

    def _run(self, text: str, gen: int) -> None:
        try:
            lang = mostly_cjk(text)
            if lang:
                # The natural voices only read English: the system's voice for the language.
                log.info("Read aloud: %s text, system voice", LANG_NAMES[lang])
                self._run_system(text, gen, lang=lang)
                return
            if self.voice.startswith(MY_VOICE):
                try:
                    self._run_piper(text, gen)
                    return
                except Exception as exc:  # noqa: BLE001 - deleted or broken voice
                    log.exception("Your voice failed; using the system voice")
                    if not self._current(gen):
                        return
                    self.on_state(f"error:Your voice couldn't play ({exc}); using the system voice")
                    self._run_system(text, gen)
                    return
            if (self.voice != "system" and not self._kokoro_failed and self._kokoro is None
                    and (self._preparing or not voice_downloaded(self.voice, self.device))):
                # First use: speak right away with the system voice while the natural one
                # downloads, instead of making you wait in silence.
                self.prepare()
                self.engine = "system"
                self.on_state("downloading")
                log.info("Read aloud: system voice for now")
                self._run_system(text, gen, announce=False)
                return
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
            self.engine = "natural"
            self.on_state("loading")
            k = Kokoro(self.device)
            with self._lock:
                self._kokoro = k
        return k

    def _run_kokoro(self, text: str, gen: int) -> None:
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
        self.engine = "natural"
        self.on_state("speaking")
        log.info("Read aloud: playing %d piece(s) with %s", len(pieces), self.voice)
        self._play(audio_q, SAMPLE_RATE, gen)

    def _run_piper(self, text: str, gen: int) -> None:
        from . import config as config_mod

        name = self.voice[len(MY_VOICE):]
        with self._lock:
            v = self._piper if self._piper and self._piper.folder.name == name else None
        if v is None:
            self.on_state("loading")
            v = PiperVoice(config_mod.config_dir() / "voices" / name)
            with self._lock:
                self._piper = v
        pieces = split_sentences(text)
        audio_q: queue.Queue = queue.Queue(maxsize=2)

        def produce():
            try:
                for sentence in pieces:
                    if not self._current(gen):
                        break
                    audio_q.put(v.synth(sentence, self.speed))
            except Exception as exc:  # noqa: BLE001
                audio_q.put(exc)
            audio_q.put(None)

        threading.Thread(target=produce, name="tts-gen", daemon=True).start()
        self.engine = "natural"
        self.on_state("speaking")
        log.info("Read aloud: playing %d sentence(s) with your voice %s", len(pieces), name)
        self._play(audio_q, v.sample_rate, gen)

    def _play(self, audio_q: queue.Queue, sample_rate: int, gen: int) -> None:
        """Play generated pieces as they arrive; pause, skip and stop react within 50 ms."""
        from .audio import output_stream

        with output_stream(samplerate=sample_rate, channels=1, dtype="float32") as out:
            while True:
                item = audio_q.get()
                if item is None or not self._current(gen):
                    break
                if isinstance(item, Exception):
                    raise item
                self._skip = False
                block = sample_rate // 20  # 50 ms: stop, pause and skip react quickly
                for i in range(0, len(item), block):
                    while self.paused and self._current(gen):
                        self.level = 0.0
                        time.sleep(0.05)
                    if not self._current(gen):
                        out.abort()
                        return
                    if self._skip:
                        break
                    chunk = item[i:i + block]
                    self.level = float(np.sqrt(np.mean(chunk * chunk))) if len(chunk) else 0.0
                    out.write(chunk)
                self.level = 0.0
                self._last_used = time.monotonic()
        self._last_used = time.monotonic()

    def _run_system(self, text: str, gen: int, announce: bool = True, lang: str | None = None) -> None:
        self.engine = "system"
        if announce:
            self.on_state("speaking")
        rate = self.speed
        if sys.platform == "darwin":
            cmd = ["say", "-r", str(int(185 * rate))]  # the text goes in on stdin
            if lang:
                voice = mac_voice_for(lang)
                if voice:
                    cmd += ["-v", voice]
                else:
                    self.on_state(f"error:No {LANG_NAMES[lang]} voice on this Mac: add one in System "
                                  "Settings > Accessibility > Spoken Content > System voice > "
                                  "Manage Voices")
        elif sys.platform == "win32":
            pick = ""
            if lang:
                culture = {"zh": "zh", "ja": "ja", "ko": "ko"}[lang]
                pick = ("$v=$s.GetInstalledVoices()|?{$_.VoiceInfo.Culture.Name -like '%s*'}|"
                        "select -First 1;if($v){$s.SelectVoice($v.VoiceInfo.Name)}"
                        "else{[Console]::Error.WriteLine('novoice')};" % culture)
            script = ("Add-Type -AssemblyName System.Speech;"
                      "$s=New-Object System.Speech.Synthesis.SpeechSynthesizer;"
                      f"$s.Rate={max(-10, min(10, round((rate - 1) * 10)))};" + pick +
                      "$s.Speak([Console]::In.ReadToEnd())")
            cmd = ["powershell", "-NoProfile", "-NonInteractive", "-Command", script]
            if lang and not _win_has_voice(lang):
                self.on_state(f"error:No {LANG_NAMES[lang]} voice on this PC: add one in Settings > "
                              "Time & language > Speech > Add voices")
        else:
            tool = shutil.which("spd-say") or shutil.which("espeak-ng") or shutil.which("espeak")
            if not tool:
                raise RuntimeError("No system voice found (install speech-dispatcher or espeak-ng)")
            cmd = [tool, "-w", text] if tool.endswith("spd-say") else [tool, text]
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL, creationflags=flags)
        self._system_proc = proc
        if sys.platform in ("win32", "darwin"):
            proc.stdin.write(text.encode("utf-8", "replace"))
        proc.stdin.close()
        t = 0.0
        while proc.poll() is None and self._current(gen):
            time.sleep(0.05)
            t += 0.05
            # No audio to measure from another process: a gentle pulse for the pill.
            self.level = 0.0 if self.paused else 0.03 + 0.02 * abs(np.sin(t * 6))
        self.level = 0.0
        self._stop_system()

    def _stop_system(self) -> None:
        proc = self._system_proc
        if proc is not None and self.paused and proc.poll() is None:
            _suspend(proc, False)  # a suspended process can't be killed cleanly everywhere
        self._stop_system_proc()

    def _stop_system_proc(self) -> None:
        proc, self._system_proc = self._system_proc, None
        if proc is not None and proc.poll() is None:
            proc.kill()

    def _unload_when_idle(self) -> None:
        while True:
            time.sleep(30)
            if ((self._kokoro is not None or self._piper is not None) and not self.speaking
                    and time.monotonic() - self._last_used > IDLE_UNLOAD_S):
                log.info("Unloading the reading voice (idle)")
                self.unload()


def _suspend(proc, pause: bool) -> None:
    """Pause or resume the system voice's process."""
    try:
        if sys.platform == "win32":
            ntdll = ctypes.windll.ntdll
            fn = ntdll.NtSuspendProcess if pause else ntdll.NtResumeProcess
            fn.argtypes = [ctypes.c_void_p]
            fn(ctypes.c_void_p(int(proc._handle)))
        else:
            import signal

            os.kill(proc.pid, signal.SIGSTOP if pause else signal.SIGCONT)
    except Exception as exc:  # noqa: BLE001
        log.debug("Couldn't %s the system voice: %s", "pause" if pause else "resume", exc)
