import pytest

from whispa.llm import needs_polish


@pytest.mark.parametrize("text", [
    "Let's meet Tuesday, no wait, Wednesday at 3.",
    "Send it to John, I mean Mike.",
    "Book the table for six, sorry, seven people.",
    "The budget is 40k, or rather 45k.",
    "Order two pizzas, no, make that three.",
    "Call Sarah, actually, call Tom instead.",
    "Can you can you send me the report?",
    "I think the the issue is the cache.",
    "We should, we should probably wait.",
    "First, buy milk. Second, call mum. Then pick up the kids.",
    "It's not Thursday but Friday.",
    "Push it to staging, wait, no, push it to prod.",
    "I'll be there at five, scratch that, six.",
    " ".join(["word"] * 3 + ["sentence"] * 0) + " " + " ".join(f"w{i}" for i in range(45)),
])
def test_flags_dictations_that_need_fixing(text):
    assert needs_polish(text) is not None


@pytest.mark.parametrize("text", [
    "Sounds good, ship it.",
    "Can you send the quarterly report to Sarah by Friday?",
    "Thanks!",
    "I actually think this is fine.",
    "What's the capital of France?",
    "That that is is a famous sentence.".replace("is is", "is"),  # "that that" is allowed
    "Let's meet Wednesday at 3 PM and loop in the platform team.",
    "No problem, see you tomorrow.",
    "Sorry I missed your call earlier.",
    "Bye bye!",
])
def test_leaves_clean_dictations_alone(text):
    assert needs_polish(text) is None
