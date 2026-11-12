"""Dictation modes: choosing one, what the AI is told, and how its answer is checked."""
import pytest

from whispa import app as app_mod
from whispa import styles
from whispa.config import Config, LLMCleanup
from whispa.focus import Focus
from whispa.llm import Polisher, check_edit


def test_every_mode_has_a_label_and_the_reshaping_ones_have_instructions():
    for mode in styles.STYLES:
        assert mode in styles.STYLE_LABELS
    for mode in ("email", "notes", "technical"):
        assert styles.prompt_for(mode)
    assert styles.prompt_for("custom", "  Keep it short ") == "Keep it short"
    assert styles.prompt_for("formal") == "" and styles.prompt_for("exact") == ""


def _app(mode="auto", custom=""):
    a = app_mod.MurmrApp.__new__(app_mod.MurmrApp)
    a.cfg = Config()
    a.cfg.mode, a.cfg.custom_mode_prompt = mode, custom
    return a


def test_picked_mode_wins_over_the_app():
    slack = Focus(app="slack")
    assert _app()._mode_for(slack) == "casual"  # automatic: chats are Message
    assert _app("email")._mode_for(slack) == "email"


def test_cycle_skips_custom_until_it_has_instructions(monkeypatch):
    for custom, expected in (("", "auto"), ("be brief", "custom")):
        a = _app("exact", custom)
        seen = []
        monkeypatch.setattr(a, "set_mode", lambda m, announce=True: seen.append(m), raising=False)
        a.set_mode = lambda m, announce=True: seen.append(m)
        a.cycle_mode()
        assert seen == [expected]


def test_check_is_relaxed_only_where_the_mode_needs_it():
    said = "okay so for the launch we need to finish the payments screen and book the review"
    notes = "- Finish the payments screen\n- Book the review"
    assert check_edit(said, notes, [], allow_drop=True) is None
    spanish = "¿Estás libre para almorzar mañana?"
    assert check_edit("hey are you free for lunch tomorrow", spanish, []) is not None
    assert check_edit("hey are you free for lunch tomorrow", spanish, [], any_words=True) is None
    essay = "Here is a poem. " * 40
    assert check_edit("write me a poem", essay, [], any_words=True) is not None


def test_mode_instructions_reach_the_model_and_translation_unlocks_language(monkeypatch):
    p = Polisher(LLMCleanup())
    msgs = p._messages("hola", "", [], styles.prompt_for("email") + "\nWrite the final text in French: x")
    assert "email" in msgs[0]["content"] and "language and meaning" not in msgs[0]["content"]
    assert len(msgs) == 2 and "Write the final text in French" in msgs[1]["content"]
    plain = p._messages("hi", "", [])
    assert len(plain) > 2  # the normal tidy-up keeps its examples


def test_mode_without_the_ai_says_so(monkeypatch):
    p = Polisher(LLMCleanup(enabled="auto"))
    monkeypatch.setattr(p, "active", lambda wait=False: False)
    assert p.polish("hi sam", mode_prompt=styles.prompt_for("email")) == "hi sam"
    assert p.last_issue == "unavailable"
    assert p.polish("hi sam") == "hi sam" and p.last_issue == ""  # plain tidy-up: no fuss


def test_reprocess_in_another_mode_keeps_your_mode(monkeypatch, tmp_path):
    from whispa.learn import Dictionary

    a = _app("casual")
    a.dictionary = Dictionary(tmp_path / "d.json")
    a.polisher = Polisher(LLMCleanup())
    seen = {}

    def polish(text, **kw):
        seen.update(kw)
        a.polisher.last_issue = ""
        return "- Finish the payments screen\n- Book the review"

    monkeypatch.setattr(a.polisher, "polish", polish)
    out = a.reprocess("um finish the payments screen and book the review", "notes")
    assert out.startswith("- Finish") and "Format it as notes" in seen["mode_prompt"]
    assert a.cfg.mode == "casual"
    assert a.reprocess("um so hi there", "exact") == "So hi there"


def test_history_row_shows_original_and_redo(tmp_path):
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from whispa.window import EntryRow

    QApplication.instance() or QApplication([])

    class Ctl:
        cfg = Config()

        def reprocess(self, raw, mode):
            return raw.upper()

    row = EntryRow({"t": 1, "raw": "um hi there", "text": "Hi there."}, lambda r: None, None, Ctl())
    assert row.original is not None and row.redo is not None
    row.original.click()
    assert row.heard.isVisibleTo(row) and "um hi there" in row.heard.text()
    same = EntryRow({"t": 1, "raw": "hi there", "text": "Hi there"}, lambda r: None)
    assert same.original is None and same.redo is None
