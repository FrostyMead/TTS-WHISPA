"""Remembering voices across meetings (opt-in)."""
import json

import numpy as np

from whispa.people import PeopleStore


def test_learn_refine_forget(tmp_path):
    store = PeopleStore(tmp_path / "people.json")
    store.learn("Priya", np.array([1.0, 0, 0]))
    store.learn("Priya", np.array([0.8, 0.6, 0]))  # refined, not replaced
    store.learn("Sam", np.array([0, 0, 2.0]))
    known = store.known()
    assert set(known) == {"Priya", "Sam"} and abs(np.linalg.norm(known["Priya"]) - 1) < 1e-5
    assert known["Priya"][0] > known["Priya"][1] > 0
    data = json.loads((tmp_path / "people.json").read_text())
    assert set(data["people"]["Priya"]) == {"v", "n", "t"} and data["people"]["Priya"]["n"] == 2
    store.forget("Sam")
    assert store.names() == ["Priya"]
    store.forget_all()
    assert store.known() == {}


def test_remembered_people_are_named_in_a_new_meeting():
    from whispa.meetings import Line, MeetingSession

    priya, sam = np.array([1.0, 0, 0]), np.array([0, 1.0, 0])
    s = MeetingSession([], lambda a, sr: "", people={"Priya": priya, "Sam": sam})
    s._trackers = {"Others": object()}
    lines = [Line(0, "Person 1", "a", 5, "Others"), Line(6, "Person 2", "b", 11, "Others")]
    s.lines = lines
    s._voiceprints = {id(lines[0]): [priya * 0.9 + 0.05], id(lines[1]): [sam]}
    s.regroup_speakers()
    assert [ln.who for ln in lines] == ["Priya", "Sam"]
    assert set(s.final_prints) == {"Priya", "Sam"}


def test_only_named_or_clearly_recognised_voices_are_learned(tmp_path, monkeypatch):
    from whispa import app as app_mod
    from whispa.config import Config

    a = app_mod.MurmrApp.__new__(app_mod.MurmrApp)
    a.cfg = Config()
    a.cfg.remember_voices = True
    store = PeopleStore(tmp_path / "people.json")
    store.learn("Ben", np.array([1.0, 0, 0]))
    monkeypatch.setattr(a, "people_store", lambda: store, raising=False)
    a.people_store = lambda: store

    class S:
        _named = {"Usman"}
        final_prints = {"Usman": np.array([0, 1.0, 0]), "Ben": np.array([0.2, 0, 1.0]),
                        "Person 3": np.array([0, 0, 1.0]), "You": np.array([1.0, 1, 0])}

    a._remember_voices("m1", S())
    assert set(store.names()) == {"Ben", "Usman"}
    assert store.known()["Ben"][0] > 0.99  # the weak "Ben" match didn't change Ben's voice
