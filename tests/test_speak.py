import pytest

from whispa.speak import split_sentences


def test_split_sentences():
    text = "Hi Sam! See https://example.com/x?y=1 for details.\n\n- First point\n- Second point"
    assert split_sentences(text) == ["Hi Sam!", "See link for details.", "First point", "Second point"]


def test_espeak_phonemes_keep_punctuation():
    pytest.importorskip("espeakng_loader")
    from whispa.speak import Espeak

    ps = Espeak().phonemize("Hello, world!")
    assert ps.startswith("h") and "," in ps and ps.endswith("!")
    assert "ˈ" in ps  # stress marks: Kokoro needs them
