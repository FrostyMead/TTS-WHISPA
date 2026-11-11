import pytest

from whispa.textproc import process


@pytest.mark.parametrize("raw, expected", [
    ("Hey, um, can you send the report to Sarah by Friday? Thanks.",
     "Hey, can you send the report to Sarah by Friday? Thanks."),
    ("Um, so I think we should ship it.", "So I think we should ship it."),
    ("It's uh really good.", "It's really good."),
    ("Done. Uh, next we deploy.", "Done. Next we deploy."),
    ("Thanks, um.", "Thanks."),
    ("Um.", ""),
    ("Umbrella and humming are fine.", "Umbrella and humming are fine."),
    ("Hmm, let me think.", "Let me think."),
])
def test_fillers(raw, expected):
    assert process(raw) == expected


def test_fillers_disabled():
    assert process("Um, hi.", fillers=False) == "Um, hi."


def test_voice_commands():
    assert process("Dear team, new line. thanks for the update.") == "Dear team,\nThanks for the update."
    assert process("First point. New paragraph. Second point.") == "First point.\n\nSecond point."


def test_replacements():
    out = process("I use open ai and chat gpt daily.",
                  replacements={"open ai": "OpenAI", "chat gpt": "ChatGPT"})
    assert out == "I use OpenAI and ChatGPT daily."


def test_replacement_word_boundaries():
    assert process("Cat concatenate.", replacements={"cat": "dog"}) == "dog concatenate."
