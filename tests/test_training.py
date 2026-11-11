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
