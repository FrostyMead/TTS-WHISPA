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


def test_starting_never_loads_voiceprints_on_the_calling_thread(tmp_path):
    """Captions / notes start from the window: setting up speakers (a model download, your
    recordings) happens on the meeting thread, so the window never freezes."""
    import threading
    import time

    from whispa.meetings import Capture, MeetingSession

    calls = []

    def slow_embedder(audio, sr):
        calls.append(threading.current_thread())
        time.sleep(0.5)
        return None

    class Mic(Capture):
        who = "You"

    (tmp_path / "clip.wav").write_bytes(b"")
    s = MeetingSession([Mic()], lambda a, sr: "", embedder=slow_embedder, profile_dir=tmp_path)
    t0 = time.perf_counter()
    s.start()
    assert time.perf_counter() - t0 < 0.3
    s.stop()
    assert all(t is not threading.main_thread() for t in calls)


def test_rename_merges_and_new_lines_use_the_name():
    from whispa.speakers import Tracker

    a, b = np.array([1.0, 0, 0]), np.array([0, 1.0, 0])
    t = Tracker("Person")
    assert t.assign(a, "x") == "Person 1" and t.assign(b, "x") == "Person 2"
    t.rename("Person 1", "Priya")
    assert t.assign(a, "x") == "Priya"
    t.rename("Person 2", "Priya")  # one person heard as two
    assert [p[0] for p in t.people] == ["Priya"]


def test_room_and_call_share_one_numbering():
    from whispa.meetings import Capture, MeetingSession

    class Mic(Capture):
        who = "You"

    class Call(Capture):
        who = "Others"

    s = MeetingSession([Mic(), Call()], lambda a, sr: "", embedder=lambda a, sr: None, room=True)
    s.channels = [type("C", (), {"capture": c})() for c in s.captures]
    s._setup_speakers()
    room, call = s._trackers["You"], s._trackers["Others"]
    assert room.assign(np.array([1.0, 0, 0]), "x") == "Person 1"
    assert call.assign(np.array([0, 1.0, 0]), "x") == "Person 2"  # not a second Person 1
    s2 = MeetingSession([Mic(), Call()], lambda a, sr: "", embedder=lambda a, sr: None)
    s2.channels = s.channels
    s2._setup_speakers()
    assert "You" not in s2._trackers  # not a room: the mic is just you


def test_live_rename_and_correction():
    from whispa.meetings import Line, MeetingSession

    s = MeetingSession([], lambda a, sr: "")
    s.lines = [Line(0, "Person 1", "the beta has forty testers"), Line(3, "Person 2", "great")]
    s.rename("Person 1", "Priya")
    assert s.lines[0].who == "Priya"
    assert s.edit_line(1, "Great, thanks.") == "great"
    assert s.lines[1].text == "Great, thanks." and s.lines[1].edited
    assert s.edit_line(9, "x") is None


def test_live_view_links_names_and_lines():
    from whispa.meetings import Line
    from whispa.meetings_page import lines_html

    out = lines_html([Line(0, "Person 1", "hello"), Line(2, "You", "hi", edited=True)], links=True,
                     edit_from=40)
    assert 'href="speaker:Person%201"' in out and 'href="line:40"' in out and 'href="line:41"' in out
    assert "edited" in out


def test_regroup_uses_long_lines_and_attaches_short_ones():
    from whispa.speakers import regroup_lines

    a, b = np.array([1.0, 0.1, 0]), np.array([0.1, 1.0, 0])
    near_b = np.array([0.4, 0.75, 0.55])  # a short, noisy line: closest to b
    groups, stats = regroup_lines([a, b, a * 0.9, b * 1.1, near_b], [5, 6, 4, 5, 1.5])
    assert groups[0] == groups[2] != groups[1] == groups[3] == groups[4]
    assert stats["closest_different"] < 0.5


def test_regroup_after_the_meeting_keeps_your_names():
    from whispa.meetings import Line, MeetingSession

    a, b = np.array([1.0, 0, 0]), np.array([0, 1.0, 0])
    s = MeetingSession([], lambda x, sr: "")
    s._trackers = {"You": object()}
    lines = [Line(0, "Person 1", "x", 5, "You"), Line(6, "Person 2", "y", 11, "You"),
             Line(12, "Person 3", "z", 17, "You"), Line(18, "Priya", "w", 23, "You")]
    s.lines = lines
    s._named = {"Priya"}
    for ln, e in zip(lines, (a, b, a, b)):  # live, Person 3 was really Person 1
        s._voiceprints[id(ln)] = [e]
    s.regroup_speakers()
    assert [ln.who for ln in lines] == ["Person 1", "Priya", "Person 1", "Priya"]


def test_clickable_lines_use_the_theme_text_colour():
    from whispa import theme
    from whispa.meetings import Line
    from whispa.meetings_page import lines_html

    out = lines_html([Line(0, "Ben", "hello")], links=True, edit_from=0)
    assert "inherit" not in out and theme.current().text in out


def test_live_transcript_stays_put_while_you_read():
    import os
    import time

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from whispa.config import Config
    from whispa.meetings import Line
    from whispa.meetings_page import MeetingsPage

    app = QApplication.instance() or QApplication([])

    class Session:
        running = True
        elapsed = 60
        lines = [Line(i, f"Person {i % 3 + 1}", f"line {i} " + "word " * 20) for i in range(60)]

        def live(self):
            return []

    session = Session()

    class Ctl:
        cfg = Config()

        def __getattr__(self, name):
            return lambda *a, **k: {"meeting_session": session, "meeting_running": True,
                                    "meetings": [], "captions_on": False}.get(name)

    page = MeetingsPage(Ctl())
    page.resize(700, 900)
    page.show()
    page._tick()
    for _ in range(5):
        app.processEvents()
    bar = page.live_view.verticalScrollBar()
    assert bar.value() == bar.maximum() and not page.live_btn.isVisibleTo(page)
    bar.setValue(bar.maximum() // 3)  # you scroll up to read
    app.processEvents()
    assert page.live_btn.isVisibleTo(page)
    where = bar.value()
    session.lines.append(Line(61, "Person 1", "someone keeps talking"))
    page._tick()
    app.processEvents()
    assert bar.value() == where  # not pulled back down
    page.live_btn.click()
    app.processEvents()
    assert bar.value() == bar.maximum() and not page.live_btn.isVisibleTo(page)
    assert "someone keeps talking" in page.live_view.toPlainText()
    page.close()
