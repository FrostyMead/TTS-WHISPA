import json

import numpy as np

from whispa.learn import Correction, TrainingStore, align_to_inserted, find_corrections


# --- labels only cover the dictated words (real cases from early recordings) ----------
def test_text_typed_after_is_dropped():
    assert align_to_inserted("Just want to taste this.", "Just want to taste this. kist") == \
        "Just want to taste this."


def test_next_dictation_pasted_behind_is_dropped():
    assert align_to_inserted("And the snippets thing?",
                             "And the snippets thing? Also is everything local at the moment?") == \
        "And the snippets thing?"


def test_text_typed_before_is_dropped():
    assert align_to_inserted("ping Sara about it", "hey team, ping Sarah about it") == "ping Sarah about it"


def test_edits_inside_are_kept():
    final = align_to_inserted("Can you ping Sara about the cooper netties cluster?",
                              "Can you ping Sarah about the Kubernetes cluster? thanks")
    assert final == "Can you ping Sarah about the Kubernetes cluster?"
    assert [(c.heard, c.meant) for c in find_corrections(
        "Can you ping Sara about the cooper netties cluster?", final)] == \
        [("Sara", "Sarah"), ("cooper netties", "Kubernetes")]


def test_punctuation_only_change_is_not_a_correction():
    assert find_corrections("so clearly it's working.", "so clearly it's working") == []


# --- store -------------------------------------------------------------------------------
def _rows(store):
    return [json.loads(x) for x in store.manifest.read_text().splitlines()]


def test_label_is_raw_transcript_plus_fixes(tmp_path):
    store = TrainingStore(tmp_path)
    sid = store.add(np.zeros(16000, np.float32), 16000, "um can you ping sara about it")
    store.record_review(sid, [Correction("Sara", "Sarah")])
    assert _rows(store)[-1] == {"id": sid, "text": "um can you ping Sarah about it", "status": "corrected"}
    assert store.stats()["corrected"] == 1


def test_unedited_is_recorded(tmp_path):
    store = TrainingStore(tmp_path)
    sid = store.add(np.zeros(16000, np.float32), 16000, "sounds good")
    store.record_review(sid, [])
    assert _rows(store)[-1] == {"id": sid, "status": "unedited"}


def test_old_polluted_labels_are_reset(tmp_path):
    (tmp_path / "manifest.jsonl").write_text(
        json.dumps({"id": "a", "audio": "a.wav", "asr": "git", "text": "git", "verified": False}) + "\n" +
        json.dumps({"id": "a", "text": "git pet", "verified": True}) + "\n")
    store = TrainingStore(tmp_path)
    rows = _rows(store)
    assert rows == [{"id": "a", "audio": "a.wav", "asr": "git", "text": "git", "status": "unknown"}]


def test_cap_deletes_oldest_uncorrected_first(tmp_path):
    store = TrainingStore(tmp_path, max_mb=1)
    ids = []
    for i in range(4):  # ~0.32 MB each
        sid = f"2026010{i}-000000-000"
        (tmp_path / f"{sid}.wav").write_bytes(b"\0" * 320_000)
        ids.append(sid)
    store._corrected.add(ids[0])  # the oldest is corrected: keep it
    store.add(np.zeros(16000, np.float32), 16000, "new")
    left = sorted(p.stem for p in tmp_path.glob("*.wav"))
    assert ids[0] in left and ids[1] not in left
    assert sum(p.stat().st_size for p in tmp_path.glob("*.wav")) <= 1_000_000


def test_delete_all(tmp_path):
    store = TrainingStore(tmp_path)
    store.add(np.zeros(16000, np.float32), 16000, "hi")
    store.delete_all()
    assert store.stats()["clips"] == 0 and not store.manifest.exists()


# --- snippets & command output ------------------------------------------------------------
def test_snippets():
    from whispa.textproc import expand_snippets

    s = {"my email": "john@example.com", "my work email": "john@corp.com", "my sign off": "Cheers,\nJohn"}
    assert expand_snippets("My email.", s) == "john@example.com"
    assert expand_snippets("Send it to my email, thanks.", s) == "Send it to john@example.com, thanks."
    assert expand_snippets("Send it to my work email.", s) == "Send it to john@corp.com."
    assert expand_snippets("My sign off.", s) == "Cheers,\nJohn"
    assert expand_snippets("I checked my emails today.", s) == "I checked my emails today."


def test_dictionary_snippets(tmp_path):
    from whispa.learn import Dictionary

    d = Dictionary(tmp_path / "dictionary.json")
    d.set_snippet("  My  Email ", "john@example.com")
    assert d.snippets_snapshot() == {"my email": "john@example.com"}
    assert Dictionary(tmp_path / "dictionary.json").snippets_snapshot() == {"my email": "john@example.com"}
    d.remove_snippet("my email")
    assert d.snippets_snapshot() == {}


def test_command_output_cleanup():
    from whispa.llm import clean_command_output as c

    assert c("Here's the revised text:\n\nDear team, see below.", "x") == "Dear team, see below."
    assert c("Sure! Here is a more formal version:\nI will be late.", "x") == "I will be late."
    assert c('"I will be ten minutes late."', None) == "I will be ten minutes late."
    assert c('"hi" and "bye"', None) == '"hi" and "bye"'
    assert c("```\nline1\nline2\n```", "x") == "line1\nline2"


def test_reading_clip_keeps_a_high_quality_copy(tmp_path):
    import json
    import wave

    import numpy as np

    from whispa.audio import resample
    from whispa.learn import HQ_SUFFIX, TrainingStore

    sr = 48000
    t = np.arange(sr * 2) / sr
    audio = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    store = TrainingStore(tmp_path)
    sid = store.add_read(audio, sr, "Hello there.", "hello there")
    main, hq = tmp_path / f"{sid}.wav", tmp_path / f"{sid}{HQ_SUFFIX}"
    with wave.open(str(main)) as w:
        assert w.getframerate() == 16000 and abs(w.getnframes() - 32000) <= 1
    with wave.open(str(hq)) as w:
        assert w.getframerate() == 24000 and abs(w.getnframes() - 48000) <= 1
    row = json.loads(store.manifest.read_text(encoding="utf-8").splitlines()[-1])
    assert row["hq"] == hq.name and row["source_sr"] == sr
    assert row["peak"] > 0.29 and row["clipped"] == 0 and row["noise_db"] < 0
    st = store.stats()
    assert st["clips"] == 1 and round(st["seconds"]) == 2  # the copy isn't counted twice
    store.remove(sid)
    assert not main.exists() and not hq.exists()
    # resampling keeps the tone (energy at 440 Hz), not just the length
    y = resample(audio, sr, 16000)
    spec = np.abs(np.fft.rfft(y))
    assert abs(np.argmax(spec) * 16000 / len(y) - 440) < 2


def test_dictation_clips_are_16k_only(tmp_path):
    import numpy as np

    from whispa.learn import HQ_SUFFIX, TrainingStore

    store = TrainingStore(tmp_path)
    store.add(np.zeros(48000, np.float32), 48000, "hi")
    assert not list(tmp_path.glob(f"*{HQ_SUFFIX}"))


def test_recorded_minutes_for_tuning(tmp_path):
    import numpy as np

    from whispa.learn import TrainingStore
    from whispa.tuning import recorded_minutes

    store = TrainingStore(tmp_path)
    sr = 48000
    store.add_read(np.zeros(sr * 30, np.float32) + 0.1, sr, "A sentence.", "a sentence")  # 30 s, HQ
    store.add_read(np.zeros(16000 * 60, np.float32) + 0.1, 16000, "Another.", "another")  # 60 s, no HQ
    sid = store.add(np.zeros(16000 * 90, np.float32), 16000, "a dictation")  # unknown: not counted
    m = recorded_minutes(tmp_path)
    assert abs(m["recognition"] - 1.5) < 0.01 and abs(m["voice"] - 0.5) < 0.01
    store.record_review(sid, [])  # left as is: still not counted ("corrected" counts)
    assert abs(recorded_minutes(tmp_path)["recognition"] - 1.5) < 0.01
