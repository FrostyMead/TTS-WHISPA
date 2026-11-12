import json

import numpy as np

from whispa.learn import (Correction, TrainingStore, align_to_inserted, edit_feedback,
                          find_corrections, review_edit, training_label)


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
    assert store.record_review(sid, "Can you ping Sara about it?",
                               "Can you ping Sarah about it?") == "corrected"
    assert _rows(store)[-1] == {"id": sid, "text": "um can you ping Sarah about it", "status": "corrected"}
    assert store.stats()["corrected"] == 1


def test_unedited_is_recorded(tmp_path):
    store = TrainingStore(tmp_path)
    sid = store.add(np.zeros(16000, np.float32), 16000, "sounds good")
    assert store.record_review(sid, "Sounds good.", "Sounds good") == "unedited"
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
    store.record_review(sid, "A dictation.", "A dictation.")  # left as is: not counted...
    assert abs(recorded_minutes(tmp_path)["recognition"] - 1.5) < 0.01
    # ...unless "Also learn from dictations I don't change" is on
    assert abs(recorded_minutes(tmp_path, include_unedited=True)["recognition"] - 3.0) < 0.01


# --- training labels follow what you actually did -----------------------------------------
def test_everyday_word_fix_is_still_a_training_label():
    """their -> there is never learned as a Dictionary rule, but the audio said "there"."""
    assert training_label("put it over their", "Put it over their.", "Put it over there.") ==         ("corrected", "put it over there")


def test_rewrite_is_kept_out_of_training():
    assert training_label("send the report to tom", "Send the report to Tom.",
                          "Can you ask Jess when she's free tomorrow?")[0] == "edited"


def test_added_or_removed_words_are_kept_out():
    assert training_label("call me back later", "Call me back later.",
                          "Call me back a bit later.")[0] == "edited"


def test_polish_that_changed_words_is_kept_out():
    """AI polish fixed a word: Parakeet's version may be wrong, so it can't be the label."""
    assert training_label("ship it to the murmer repo", "Ship it to the Murmr repo.",
                          "Ship it to the Murmr repo.")[0] == "polished"


def test_polish_dropping_fillers_or_adding_a_sign_off_is_fine():
    signed = "So the build passed.\n\nThanks,"
    assert training_label("um so the build passed", signed, signed) == \
        ("unedited", "um so the build passed")


def test_ambiguous_fix_is_kept_out():
    """Two "the"s in what was heard: we can't tell which one you changed."""
    assert training_label("the cat and the dog", "The cat and the dog.", "The cat and a dog.")[0] == "edited"


# --- what the pill says after you fix something ------------------------------------------
def test_feedback_lines():
    r = review_edit("ping Sara", "ping Sarah")
    assert edit_feedback(r, [Correction("Sara", "Sarah")], [], "corrected") ==         "Learned: Sarah · saved for voice training"
    r = review_edit("the deploy step", "the deployed step")
    assert edit_feedback(r, [], [Correction("deploy", "deployed")], "") ==         "Got it: deploy → deployed (fix it once more to learn it)"
    r = review_edit("over their", "over there")
    assert edit_feedback(r, [], [], "corrected") == "Saved your fix for voice training"
    assert "not a mishearing" in edit_feedback(r, [], [], "")
    assert edit_feedback(review_edit("call me later", "call me a bit later"), [], [], "edited") ==         "You added or removed words, so nothing was learned"
    assert "rewrite" in edit_feedback(review_edit("send it to tom", "ask Jess about Friday"), [], [], "")
    assert edit_feedback(review_edit("Sounds good.", "Sounds good"), [], [], "unedited") is None


def test_watcher_reports_a_fix(tmp_path):
    from whispa.learn import CorrectionWatcher, Dictionary

    class F:
        readable = True
        text = "hi ping Sara now"

        def read_text(self):
            return self.text

    said = []
    reviewed = []
    w = CorrectionWatcher(Dictionary(tmp_path / "d.json"), on_feedback=said.append,
                          on_final=lambda sid, a, b: reviewed.append((sid, a, b)) or "corrected")
    w.watch(F(), "hi  now", "ping Sara", "s1")
    f = w._session["focus"]
    f.text = "hi ping Sarah now"
    w._poll(w._session)
    w.finish()
    assert reviewed == [("s1", "ping Sara", "ping Sarah")]
    assert said == ["Learned: Sarah · saved for voice training"]
