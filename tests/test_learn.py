import numpy as np

from whispa.learn import (Correction, Dictionary, TrainingStore, edited_region, find_corrections,
                          needs_confirmation)
from whispa.textproc import process


def test_edited_region_insert_in_middle():
    before = "Hello team, \n\nThanks"
    after = "Hello team, Can you ping Sarah? \n\nThanks"
    assert edited_region(before, after).strip() == "Can you ping Sarah?"


def test_edited_region_empty_field():
    assert edited_region("", "Ping Sarah about it.") == "Ping Sarah about it."


def test_finds_misheard_name():
    cs = find_corrections("Can you ping Sara about the cooper netties cluster?",
                          "Can you ping Sarah about the Kubernetes cluster?")
    assert [(c.heard, c.meant) for c in cs] == [("Sara", "Sarah"), ("cooper netties", "Kubernetes")]


def test_ignores_content_edits():
    assert find_corrections("Let's meet Tuesday at noon.", "Let's meet Wednesday at noon.") == []
    assert find_corrections("I think their right.", "I think there right.") == []


def test_ignores_rewrites():
    assert find_corrections("Can you send the report", "Never mind, I'll do it myself tomorrow") == []


def test_ignores_apostrophe_style():
    assert find_corrections("Don't worry about it", "Don’t worry about it") == []


def test_case_fix_of_brand_learned_immediately():
    cs = find_corrections("I pushed it to github already", "I pushed it to GitHub already")
    assert [(c.heard, c.meant) for c in cs] == [("github", "GitHub")]
    assert not needs_confirmation(cs[0])


def test_sentence_start_capitalisation_needs_confirmation():
    assert needs_confirmation(Correction("paris", "Paris")) is True


def test_dictionary_learns_and_applies(tmp_path):
    d = Dictionary(tmp_path / "dictionary.json")
    assert d.learn(Correction("Sara", "Sarah"))
    terms, corrections = d.snapshot()
    assert corrections == {"sara": "Sarah"} and "Sarah" in terms
    assert process("Tell Sara hi.", replacements=corrections) == "Tell Sarah hi."
    # persisted
    d2 = Dictionary(tmp_path / "dictionary.json")
    assert d2.snapshot()[1] == {"sara": "Sarah"}


def test_common_word_fix_needs_two_sightings(tmp_path):
    d = Dictionary(tmp_path / "dictionary.json")
    c = Correction("rite", "write")
    assert not d.learn(c)
    assert d.snapshot()[1] == {}
    assert d.learn(c)
    assert d.snapshot()[1] == {"rite": "write"}


def test_dictionary_picks_up_hand_edits(tmp_path):
    import json, os, time

    path = tmp_path / "dictionary.json"
    d = Dictionary(path)
    data = json.loads(path.read_text())
    data["terms"] = ["Whispa"]
    path.write_text(json.dumps(data))
    os.utime(path, (time.time() + 5, time.time() + 5))
    assert d.snapshot()[0] == ["Whispa"]


def test_scratch_that_and_terms():
    assert process("Send it Friday. Scratch that. Send it Monday.") == "Send it Monday."
    assert process("We deploy on kubernetis.", terms=["Kubernetes"]) == "We deploy on Kubernetes."
    assert process("The data is fine.", terms=["Dana"]) == "The data is fine."
