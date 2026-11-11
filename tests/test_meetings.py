import numpy as np

from whispa import meetings
from whispa.llm import parse_summary, split_transcript
from whispa.meetings import Capture, Line, MeetingSession, MeetingStore, find_pause, is_silent

SR = 16000


def speech(seconds, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(int(seconds * SR)) * 0.2).astype(np.float32)


def silence(seconds):
    return np.zeros(int(seconds * SR), np.float32)


class FakeCapture(Capture):
    def __init__(self, who="You"):
        super().__init__()
        self.who = who
        self.samplerate = SR


def feed(session, cap, audio, step_s=0.5):
    """Push audio in real-time-sized pieces, running the session's step after each."""
    n = int(step_s * SR)
    for i in range(0, len(audio), n):
        cap._feed(audio[i:i + n])
        for ch in session.channels:
            session._step(ch)


def make(caps, texts=None):
    calls = []

    def transcribe(audio, sr):
        calls.append(len(audio) / sr)
        return texts(len(calls)) if texts else f"Line {len(calls)}."

    s = MeetingSession(caps, transcribe)
    return s, calls


def test_pause_detection():
    assert find_pause(np.concatenate([speech(2.5), silence(0.6), speech(0.5)]), SR) is not None
    assert find_pause(speech(4), SR) is None  # no pause
    assert find_pause(np.concatenate([speech(1.0), silence(0.6)]), SR) is None  # too short a line
    assert is_silent(silence(2), SR) and not is_silent(speech(1), SR)


def test_lines_end_at_pauses_and_audio_is_dropped():
    cap = FakeCapture()
    s, calls = make([cap])
    feed(s, cap, np.concatenate([speech(3), silence(0.8), speech(3, 1), silence(2.5)]))
    finals = [line.text for line in s.lines]
    assert len(finals) == 2, finals
    assert cap.buffer._start > 6 * SR  # finished audio isn't kept
    assert all(c <= 4.5 for c in calls)  # never re-transcribes the whole meeting


def test_live_text_appears_before_the_line_is_final():
    cap = FakeCapture()
    s, _ = make([cap])
    feed(s, cap, speech(2.0))
    assert s.live() and not s.lines


def test_long_speech_without_pauses_is_split():
    cap = FakeCapture()
    s, _ = make([cap])
    feed(s, cap, speech(meetings.MAX_LINE_S + 3))
    assert len(s.lines) >= 1


def test_echo_of_the_other_people_is_dropped():
    you, others = FakeCapture("You"), FakeCapture("Others")
    s, _ = make([you, others], texts=lambda n: "Can everyone hear me okay")
    s.channels = [s.channels[1], s.channels[0]]  # computer audio first, then the mic's copy
    a = np.concatenate([speech(3), silence(1.0)])
    feed(s, others, a)
    feed(s, you, a)
    assert [line.who for line in s.lines] == ["Others"]


def test_echo_that_arrives_first_is_replaced():
    you, others = FakeCapture("You"), FakeCapture("Others")
    s, _ = make([you, others], texts=lambda n: "Let's start with the budget")
    a = np.concatenate([speech(3), silence(1.0)])
    feed(s, you, a)
    feed(s, others, a)
    assert [line.who for line in s.lines] == ["Others"]


def test_store_roundtrip(tmp_path):
    store = MeetingStore(tmp_path)
    mid = store.save({"started": 1.0, "duration": 65, "lines": [Line(0.0, "You", "hello there").__dict__],
                      "title": "Kickoff", "summary": "## Summary\n- hi"})
    assert store.load(mid)["title"] == "Kickoff"
    listed = store.list()
    assert listed[0]["id"] == mid and listed[0]["words"] == 2
    assert "hello there" in meetings.to_markdown(store.load(mid))
    store.delete(mid)
    assert store.list() == []


def test_summary_parsing_and_chunking():
    title, body = parse_summary("Title: **Q3 budget review**\n## Summary\n- Spend is down\n## Decisions\nNone.")
    assert title == "Q3 budget review" and body.startswith("## Summary")
    text = "\n".join(f"[0:{i:02d}] You: " + "word " * 50 for i in range(60))
    chunks = split_transcript(text, size=1000)
    assert len(chunks) > 1 and all(len(c) <= 1300 for c in chunks)
    assert "\n".join(chunks) == text


def test_a_sentence_split_at_a_short_pause_is_one_line():
    cap = FakeCapture()
    s, _ = make([cap])
    ch = s.channels[0]
    s._add(ch, 0, "I think we can ship on the 20th", 3 * SR)
    s._add(ch, int(3.7 * SR), "if the testing goes well.", 3 * SR)  # 0.7 s later
    s._add(ch, int(7.4 * SR), "Next topic.", 2 * SR)  # after a full stop: a new line
    s._add(ch, int(20 * SR), "much later", 1 * SR)  # long gap: a new line
    assert [line.text for line in s.lines] == [
        "I think we can ship on the 20th if the testing goes well.", "Next topic.", "much later"]


# --------------------------------------------------------------------------- speakers
def voice(seconds, level, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(int(seconds * SR)) * level).astype(np.float32)


def fake_embedder(audio, sr):
    """Two 'voices' told apart by loudness (stand-in for real voiceprints)."""
    if len(audio) < 1.2 * sr:
        return None
    rms = float(np.sqrt(np.mean(audio ** 2)))
    v = np.array([1.0, 0.0]) if rms > 0.15 else np.array([0.0, 1.0])
    return v


def test_people_on_the_call_are_told_apart():
    others = FakeCapture("Others")
    s, _ = make([others])
    s.embedder = fake_embedder
    s._setup_speakers()
    gap = silence(1.0)
    feed(s, others, np.concatenate([voice(3, 0.2), gap, voice(3, 0.08, 1), gap, voice(3, 0.2, 2), gap,
                                    silence(1.5)]))
    assert [line.who for line in s.lines] == ["Person 1", "Person 2", "Person 1"]
    assert all(line.channel == "Others" for line in s.lines)


def test_mic_only_meeting_recognises_you(tmp_path):
    you = FakeCapture("You")
    s, _ = make([you])
    s.embedder = fake_embedder
    s._setup_speakers()
    s._trackers["You"].known["You"] = np.array([1.0, 0.0])  # your voice profile
    gap = silence(1.0)
    feed(s, you, np.concatenate([voice(3, 0.2), gap, voice(3, 0.08, 1), gap, silence(1.5)]))
    assert [line.who for line in s.lines] == ["You", "Person 1"]


def test_rename_speaker_everywhere(tmp_path, monkeypatch):
    from whispa import app as appmod

    store = MeetingStore(tmp_path)
    mid = store.save({"started": 1.0, "duration": 10, "title": "Plan with Person 2",
                      "summary": "- Person 2 will send the deck. Person 12 is someone else.",
                      "lines": [{"t": 0, "who": "Person 2", "text": "I'll send it."},
                                {"t": 5, "who": "Person 3", "text": "Thanks."}]})
    fake = type("A", (), {"meeting_store": lambda self: store})()
    appmod.MurmrApp.rename_speaker(fake, mid, "Person 2", "Sarah")
    m = store.load(mid)
    assert [line["who"] for line in m["lines"]] == ["Sarah", "Person 3"]
    assert m["summary"] == "- Sarah will send the deck. Person 12 is someone else."
    assert m["title"] == "Plan with Sarah"
    appmod.MurmrApp.rename_speaker(fake, mid, "Person 3", "Sarah")  # one person heard as two: merge
    assert {line["who"] for line in store.load(mid)["lines"]} == {"Sarah"}


def test_action_items_get_exactly_one_checkbox():
    _, body = parse_summary("Title: x\n## Action items\n- [ ] - [ ] Person 1: Book the review.\n"
                            "* [x] You: send notes\n- Sarah: call the vendor\n- Others: None.\n## Open questions\nNone.")
    items = [line for line in body.splitlines() if line.startswith("- [")]
    assert items == ["- [ ] Person 1: Book the review.", "- [ ] You: send notes",
                     "- [ ] Sarah: call the vendor"]
