"""Entry point: ``python -m whispa`` runs the app; flags give diagnostics."""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="whispa", description="Local push-to-talk dictation.")
    parser.add_argument("--transcribe", metavar="WAV", help="transcribe a WAV file and exit")
    parser.add_argument("--list-devices", action="store_true", help="list microphones and exit")
    parser.add_argument("--test-mic", action="store_true", help="record 3 s and report the level")
    parser.add_argument("--config-path", action="store_true", help="print the settings file path")
    parser.add_argument("--autostart", choices=["on", "off"], help="enable/disable launch at login")
    parser.add_argument("--background", action="store_true",
                        help="start in the tray without opening the window")
    parser.add_argument("--install-shortcuts", action="store_true",
                        help="add Whispa to the Start Menu and Desktop (macOS: ~/Applications/Whispa.app)")
    parser.add_argument("--download-model", action="store_true",
                        help="download and test-load the speech model, then exit")
    args = parser.parse_args(argv)

    if args.config_path:
        from . import config

        config.load()
        print(config.config_path())
        return 0
    if args.list_devices:
        from .audio import list_input_devices

        print("\n".join(list_input_devices()))
        return 0
    if args.test_mic:
        from . import config
        from .audio import test_mic

        print("Speak now…")
        print(test_mic(config.load().input_device))
        return 0
    if args.autostart:
        from . import autostart

        autostart.set_enabled(args.autostart == "on")
        print(f"Launch at login: {args.autostart}")
        return 0
    if args.install_shortcuts:
        from .shortcuts import install

        for path in install():
            print(f"Created {path}")
        return 0
    if args.download_model:
        from . import config
        from .asr import Transcriber

        cfg = config.load()
        t = Transcriber(cfg.model, cfg.device, cfg.quantization)
        t.load()
        print(f"Speech model ready ({t.backend})")
        return 0
    if args.transcribe:
        return _transcribe(args.transcribe)

    from .app import run

    return run(background=args.background)


def _transcribe(path: str) -> int:
    import time
    import wave

    import numpy as np

    from . import config, textproc
    from .asr import Transcriber

    cfg = config.load()
    with wave.open(path) as w:
        if w.getsampwidth() != 2:
            print("Only 16-bit PCM WAV is supported here", file=sys.stderr)
            return 2
        sr, ch = w.getframerate(), w.getnchannels()
        audio = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
    if ch > 1:
        audio = audio.reshape(-1, ch).mean(axis=1)
    tr = Transcriber(cfg.model, cfg.device, cfg.quantization)
    t0 = time.perf_counter()
    tr.load()
    print(f"[loaded {cfg.model} on {tr.backend} in {time.perf_counter() - t0:.1f}s]", file=sys.stderr)
    t0 = time.perf_counter()
    raw = tr.transcribe(audio, sr)
    ms = (time.perf_counter() - t0) * 1000
    text = textproc.process(raw, fillers=cfg.remove_fillers, commands=cfg.voice_commands,
                            replacements=cfg.replacements)
    print(f"[{len(audio) / sr:.1f}s audio in {ms:.0f} ms]", file=sys.stderr)
    print(f"raw:     {raw}\ncleaned: {text}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
