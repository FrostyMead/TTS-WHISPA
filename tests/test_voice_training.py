import json
import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from whispa.config import Config  # noqa: E402
from whispa.learn import Dictionary, TrainingStore  # noqa: E402
from whispa.voice_training import (PromptQueue, VoiceTrainingPage, builtin_sentences,  # noqa: E402
                                   check_take, dictionary_sentences, dictionary_words,
                                   normalize_words, word_error_rate)

SR = 16000


def speech(seconds=2.0, amp=0.1):
    t = np.arange(int(seconds * SR)) / SR
    return (amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


# --- checking a take ---------------------------------------------------------------------
def test_number_and_punctuation_style_dont_count():
    assert normalize_words("Meet me at 3:30 PM, on May 4.") == normalize_words("meet me at 3.30 p.m. on May 4")
    assert word_error_rate("We need two tables.", "We need 2 tables") == 0
    assert word_error_rate("Sales rose 25% in Q3.", "Sales rose 25 percent in Q3.") == 0
    assert word_error_rate("It's done.", "its done") == 0


def test_wer():
    assert word_error_rate("one two three four", "one two four") == pytest.approx(0.25)
    assert word_error_rate("a b", "") == 1


def test_good_take_passes():
    v = check_take("Can you ping Sarah about the Kubernetes cluster?",
                   "Can you ping Sara about the Kubernetes cluster?", speech(), SR)
    assert v.ok and v.wer == pytest.approx(1 / 8)


def test_misread_is_flagged():
    v = check_take("Can you ping Sarah about the Kubernetes cluster?",
                   "I'll be there in ten minutes.", speech(), SR)
    assert not v.ok and "different" in v.reason


def test_silence_and_short_takes_are_flagged():
    assert "microphone" in check_take("Hello there everyone.", "", np.zeros(SR * 2, np.float32), SR).reason
    assert "too short" in check_take("Hello there everyone.", "hello", speech(0.3), SR).reason
    assert not check_take("Hello there everyone.", "", speech(), SR).ok


# --- sentences ------------------------------------------------------------------------------
def test_builtin_set_is_clean():
    sentences = builtin_sentences()
    assert len(sentences) >= 300
    assert len(set(sentences)) == len(sentences)
    for s in sentences:
        assert s[-1] in ".?!", s
        assert not any(c in s for c in "/&#@()[];…\n"), s
        assert 6 <= len(s.split()) <= 30, s


def test_dictionary_sentences(tmp_path):
    d = Dictionary(tmp_path / "dictionary.json")
    d.add_term("Kubernetes")
    d.set_correction("cooper netties", "Kubernetes")  # same word: listed once
    d.set_correction("sara", "Sarah")
    words = dictionary_words(d)
    assert words == ["Kubernetes", "Sarah"]
    out = dictionary_sentences(words, per_word=3)
    assert len(out) == 6 and sum("Sarah" in s for s in out) == 3
    assert dictionary_sentences(words) == out  # stable between sessions
    assert all(s[0].isupper() for s in out)


def test_queue_mixes_dictionary_and_skips_done():
    builtin = [f"Sentence number {i} is here." for i in range(10)]
    q = PromptQueue(builtin, ["Mine A.", "Mine B."], done={builtin[0], "Mine B."})
    texts = [p.text for p in q.items]
    assert builtin[0] not in texts and "Mine B." not in texts
    assert len(texts) == 10  # 9 built-in + 1 dictionary
    assert q.items[0].source == "dictionary"
    first = q.current
    q.skip()
    assert q.items[-1] is first
    q.advance()
    assert len(q) == 9


def test_queue_order_is_stable_and_starts_over_when_done():
    builtin = [f"Sentence number {i} is here." for i in range(10)]
    a = [p.text for p in PromptQueue(builtin, [], set()).items]
    assert a == [p.text for p in PromptQueue(builtin, [], set()).items]
    q = PromptQueue(builtin, [], set(builtin))
    assert q.round == 2 and len(q) == 10


# --- store ----------------------------------------------------------------------------------
def _rows(store):
    return [json.loads(x) for x in store.manifest.read_text(encoding="utf-8").splitlines()]


def test_read_clips(tmp_path):
    store = TrainingStore(tmp_path)
    a = store.add_read(speech(2.0), SR, "Exact prompt text.", "exact prompt text")
    b = store.add_read(speech(3.0), SR, "Second one.", "second one")
    assert a != b
    row = _rows(store)[0]
    assert row["status"] == "read" and row["text"] == "Exact prompt text." and row["asr"] == "exact prompt text"
    assert store.read_seconds() == pytest.approx(5.0)
    assert store.read_prompts() == {"Exact prompt text.", "Second one."}
    store.remove(b)
    assert not (tmp_path / f"{b}.wav").exists()
    assert _rows(store)[-1] == {"id": b, "status": "deleted"}
    reloaded = TrainingStore(tmp_path)
    assert reloaded.read_prompts() == {"Exact prompt text."}
    assert reloaded.stats()["read"] == 1


def test_cap_keeps_read_clips(tmp_path):
    store = TrainingStore(tmp_path, max_mb=1)
    sid = store.add_read(speech(10.0), SR, "Keep me.", "keep me")  # 0.32 MB
    for i in range(3):
        (tmp_path / f"2000010{i}-000000-000.wav").write_bytes(b"\0" * 320_000)
    store.add(speech(1.0), SR, "new")
    assert (tmp_path / f"{sid}.wav").exists()


def test_legacy_reset_keeps_read_labels(tmp_path):
    (tmp_path / "manifest.jsonl").write_text(
        json.dumps({"id": "a", "audio": "a.wav", "asr": "git", "text": "git", "verified": False}) + "\n" +
        json.dumps({"id": "b", "audio": "b.wav", "asr": "hi", "text": "Hi.", "status": "read"}) + "\n")
    rows = _rows(TrainingStore(tmp_path))
    assert [r["status"] for r in rows] == ["unknown", "read"]


# --- the page -------------------------------------------------------------------------------
class FakeRecorder:
    def __init__(self):
        self.level = 0.05
        self.next = speech()

    def start(self):
        pass

    def stop(self):
        return self.next, SR


class FakeController:
    def __init__(self, tmp_path):
        self.cfg = Config()
        self.cfg.save_training_audio = True  # opted in (the page needs it)
        self.dictionary = Dictionary(tmp_path / "dictionary.json")
        self.dictionary.add_term("Siobhan")
        self.store = TrainingStore(tmp_path / "training")
        self.heard = None  # None = echo the prompt

    def training_store(self):
        return self.store

    def transcribe_sample(self, audio, sr):
        return self.heard if self.heard is not None else self.page._queue.current.text


@pytest.fixture
def page(tmp_path):
    app = QApplication.instance() or QApplication([])
    ctl = FakeController(tmp_path)
    rec = FakeRecorder()
    p = VoiceTrainingPage(ctl, recorder_factory=lambda: rec)
    ctl.page = p
    p.resize(900, 700)
    p.refresh()
    p.rec, p.fake, p.app = rec, ctl, app
    return p


def _wait(page, state="checking", timeout=5.0):
    end = time.monotonic() + timeout
    while page._state == state and time.monotonic() < end:
        page.app.processEvents()
        time.sleep(0.01)


def test_page_saves_good_take_and_advances(page):
    first = page._queue.current.text
    assert first in page.sentence.text() and page._queue.current.source == "dictionary"
    page.start_take()
    assert page._state == "recording"
    page.stop_take()
    _wait(page)
    assert page._state == "idle"
    rows = _rows(page.fake.store)
    assert rows[-1]["text"] == first and rows[-1]["status"] == "read"
    assert page._queue.current.text != first
    assert "Saved" in page.result_title.text()
    assert page.fake.store.read_seconds() == pytest.approx(2.0)

    page.redo_last()  # re-record: clip removed, sentence back on screen
    assert page._queue.current.text == first
    assert page.fake.store.read_seconds() == 0


def test_page_flags_misread_then_save_anyway(page):
    page.fake.heard = "something else entirely and not at all close"
    first = page._queue.current.text
    page.start_take()
    page.stop_take()
    _wait(page)
    assert page._state == "flagged"
    assert not page.fake.store.manifest.exists()
    assert page.save_anyway.isVisibleTo(page) and page.retry.isVisibleTo(page)
    page._save_anyway()
    assert _rows(page.fake.store)[-1]["text"] == first


def test_page_flags_silence_without_saving(page):
    page.rec.next = np.zeros(SR * 2, np.float32)
    page.start_take()
    page.stop_take()
    _wait(page)
    assert page._state == "flagged" and "microphone" in page.result_title.text()
    assert not page.save_anyway.isVisibleTo(page)
    assert not page.fake.store.manifest.exists()


def test_skip(page):
    first = page._queue.current.text
    page.skip_prompt()
    assert page._queue.current.text != first and page._queue.items[-1].text == first


def test_page_records_nothing_without_consent(tmp_path):
    app = QApplication.instance() or QApplication([])
    ctl = FakeController(tmp_path)
    ctl.cfg.save_training_audio = False
    ctl.cfg.voice_clone = False
    rec = FakeRecorder()
    p = VoiceTrainingPage(ctl, recorder_factory=lambda: rec)
    ctl.page = p
    p.refresh()
    assert not p.record.isEnabled() and "Your voice" in p.sentence.text()
    p.start_take()  # e.g. holding Space
    assert p._state == "idle" and not ctl.store.manifest.exists()  # nothing saved
    ctl.cfg.voice_clone = True  # either opt-in is enough
    p.refresh()
    assert p.record.isEnabled()


# --- paragraphs (faster) ---------------------------------------------------------------------
def test_passages_are_clean_and_short_enough_to_train_on():
    from whispa.voice_training import builtin_passages

    passages = builtin_passages()
    assert len(passages) >= 80 and len(set(passages)) == len(passages)
    for p in passages:
        assert p[-1] in ".?!", p
        assert not any(c in p for c in "/&#@()[];…\n"), p
        assert 20 <= len(p.split()) <= 36, p  # ~15 s: under the 20 s training limit


def test_join_sentences_respects_limits():
    from whispa.voice_training import join_sentences

    sents = [f"Word {i} " + "x " * 8 + "end." for i in range(10)]  # 11 words each
    out = join_sentences(sents, max_words=34, per=3)
    assert all(len(p.split()) <= 34 for p in out)
    assert " ".join(out).split() == " ".join(sents).split()  # nothing lost or reordered


def test_paragraph_prompts_skip_sentences_already_read():
    from whispa.voice_training import builtin_passages, paragraph_prompts

    sents = builtin_sentences()
    first = paragraph_prompts(sents, set())
    joined = first[len(builtin_passages())]
    single = sents[5]
    again = paragraph_prompts(sents, {joined, single})
    rest = " ".join(again)
    for s in sents:
        if s in joined or s == single:
            assert s not in rest
    assert len(again) < len(first)


def test_too_long_take_is_flagged_without_save():
    v = check_take("Some paragraph.", "some paragraph", speech(21.0), SR)
    assert not v.ok and "20" in v.reason


def test_page_paragraph_mode_and_switching(page):
    assert page._paragraphs() and len(page._queue.current.text.split()) > 15
    page.length.setCurrentIndex(1)  # single sentences
    assert page.fake.cfg.training_prompts == "sentences"
    assert len(page._queue.current.text.split()) <= 30
    page.length.setCurrentIndex(0)
    assert page.fake.cfg.training_prompts == "paragraphs"
