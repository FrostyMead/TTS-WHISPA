"""Computer commands: only clear, explicit phrases become actions."""
import pytest

from whispa import commands
from whispa.commands import Action, best_match, parse


@pytest.mark.parametrize("said,expected", [
    ("Open Spotify.", Action("open_app", "spotify")),
    ("please open up the calculator app", Action("open_app", "calculator")),
    ("switch to Chrome", Action("switch_app", "chrome")),
    ("go to my Outlook window", Action("switch_app", "outlook")),
    ("open downloads", Action("open_folder", "Downloads")),
    ("open my documents folder", Action("open_folder", "Documents")),
    ("open youtube dot com", Action("open_url", "youtube.com")),
    ("open bbc.co.uk", Action("open_url", "bbc.co.uk")),
    ("turn the volume down", Action("volume", "down")),
    ("louder", Action("volume", "up")),
    ("mute", Action("volume", "mute")),
])
def test_clear_phrases_become_actions(said, expected):
    assert parse(said) == expected


@pytest.mark.parametrize("said", [
    "make this more formal", "translate to Spanish", "write a thank you note to Sam",
    "open the report I sent you yesterday and summarise it", "delete all my files",
    "send this to the team", "",
])
def test_everything_else_is_left_to_command_mode(said):
    assert parse(said) is None


def test_app_names_match_what_was_said():
    apps = {"Spotify": "a", "Google Chrome": "b", "Microsoft Outlook": "c", "Calculator": "d"}
    assert best_match("spotify", apps) == "Spotify"
    assert best_match("chrome", apps) == "Google Chrome"
    assert best_match("google chrome", apps) == "Google Chrome"
    assert best_match("calculater", apps) == "Calculator"
    assert best_match("photoshop", apps) is None


def test_unknown_app_is_a_clear_message(monkeypatch):
    monkeypatch.setattr(commands, "installed_apps", lambda: {"Spotify": "x"})
    monkeypatch.setattr(commands, "_bring_to_front", lambda name: False)
    with pytest.raises(RuntimeError, match="Couldn't find an app"):
        commands.run(Action("open_app", "photoshop"))
    opened = []
    monkeypatch.setattr(commands, "_open_path", opened.append)
    assert commands.run(Action("open_app", "spotify")) == "Opening Spotify" and opened == ["x"]


def test_a_different_app_is_never_guessed():
    apps = {"Steam": "a", "TeamViewer": "b", "Calculator": "c"}
    assert best_match("teams", apps) is None
    assert best_match("calculater", apps) == "Calculator"
