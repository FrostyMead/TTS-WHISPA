"""Transcribing a file: decoding (PyAV), one speaker per piece, and saving it as a meeting."""
import wave

import numpy as np
import pytest

from whispa import filetrans


def _tone_wav(path, pieces):
    """Each piece: (seconds of 'speech' tone at a pitch, seconds of silence after)."""
    sr = 16000
    parts = []
    for secs, pitch, gap in pieces:
        t = np.arange(int(secs * sr)) / sr
        parts += [0.3 * np.sin(2 * np.pi * pitch * t), np.zeros(int(gap * sr))]
    audio = np.concatenate(parts).astype(np.float32)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((audio * 32767).astype(np.int16).tobytes())
    return audio


def test_decode_reads_the_whole_file_at_16k(tmp_path):
    audio = _tone_wav(tmp_path / "a.wav", [(3, 220, 0.5)])
    got = np.concatenate(list(filetrans.decode(tmp_path / "a.wav", block_s=1)))
    assert abs(len(got) - len(audio)) < 400
    assert filetrans.duration_s(tmp_path / "a.wav") == pytest.approx(3.5, abs=0.1)


def test_one_piece_per_turn_with_times(tmp_path):
    _tone_wav(tmp_path / "talk.wav", [(2.5, 220, 1.0), (3.0, 440, 1.0), (2.0, 220, 0.5)])
    pieces = []

    def transcribe(audio, sr):
        pieces.append(len(audio) / sr)
        return f"piece {len(pieces)}"

    voices = {}

    def embedder(audio, sr):  # the "voice" is the tone's pitch
        spec = np.abs(np.fft.rfft(audio[:sr]))
        v = np.zeros(4)
        v[0 if np.argmax(spec) < 330 else 1] = 1.0
        return v

    lines = filetrans.transcribe_file(tmp_path / "talk.wav", transcribe, embedder=embedder)
    assert [l.text for l in lines] == ["piece 1", "piece 2", "piece 3"]
    assert lines[0].who == lines[2].who != lines[1].who
    assert lines[1].t == pytest.approx(3.0, abs=0.4)
    m = filetrans.meeting_from_file(tmp_path / "talk.wav", lines, 1.0)
    assert m["title"] == "talk" and m["sources"] == ["File"] and len(m["lines"]) == 3


def test_cancel_stops_it(tmp_path):
    import threading

    _tone_wav(tmp_path / "long.wav", [(20, 220, 0.5), (20, 220, 0.5)])
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(filetrans.Cancelled):
        filetrans.transcribe_file(tmp_path / "long.wav", lambda a, sr: "x", cancel=cancel)
