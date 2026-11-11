import pytest

from whispa.formatting import smart_format
from whispa.textproc import process


@pytest.mark.parametrize("spoken, written", [
    # emails
    ("Send it to john dot smith at gmail dot com please.", "Send it to john.smith@gmail.com please."),
    ("My email is Angus at acme dot co dot nz.", "My email is angus@acme.co.nz."),
    ("Reach me at sam underscore lee at outlook.com", "Reach me at sam_lee@outlook.com"),
    ("Write to jo at example.com.", "Write to jo@example.com."),
    ("already jo@example.com here", "already jo@example.com here"),
    # websites
    ("Go to whispa dot io slash docs for more.", "Go to whispa.io/docs for more."),
    ("Check example dot com.", "Check example.com."),
    ("It's on www dot python dot org", "It's on www.python.org"),
    ("The dot com bubble burst.", "The dot com bubble burst."),
    ("I'm at the station.", "I'm at the station."),
    ("We met at Google in 2020.", "We met at Google in 2020."),
    # money
    ("It costs 20 dollars.", "It costs $20."),
    ("That's $20 dollars total.", "That's $20 total."),
    ("About 10 euros and 50 cents.", "About €10.50."),
    ("1,500 dollars a month", "$1,500 a month"),
    ("I lost 5 pounds.", "I lost 5 pounds."),
    # percent
    ("Up 50 percent this year.", "Up 50% this year."),
    ("A 12.5 per cent rise", "A 12.5% rise"),
    # times
    ("See you at 330 pm.", "See you at 3:30 pm."),
    ("Call at 3.30 p.m. tomorrow", "Call at 3:30 pm tomorrow"),
    ("Meet at 5pm", "Meet at 5 pm"),
    ("Starts at 10 30 am.", "Starts at 10:30 am."),
    ("Room 2025 am I right", "Room 2025 am I right"),
])
def test_smart_format(spoken, written):
    assert smart_format(spoken) == written


def test_bullet_points():
    out = process("Groceries: bullet point milk, bullet point eggs, bullet point bread.")
    assert out == "Groceries:\n- Milk\n- Eggs\n- Bread."
    assert process("Bullet point first thing. Bullet point second thing.") == \
        "- First thing.\n- Second thing."
    assert process("Bite the bullet and ship it.") == "Bite the bullet and ship it."


def test_smart_is_optional():
    assert process("It costs 20 dollars.") == "It costs 20 dollars."
    assert process("It costs 20 dollars.", smart=True) == "It costs $20."
