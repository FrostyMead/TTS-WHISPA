"""Deterministic clean-up applied to raw transcripts."""

from __future__ import annotations

import re

_CAP = "\x00"  # marks a position whose next letter should be capitalized

_FILLER = r"(?:u+m+|u+h+m*|e+r+m+|h+m+|m+h+m+)"
_FILLER_RE = re.compile(rf"(?<![\w'-]){_FILLER}(?![\w'-])[,…]*\s*", re.IGNORECASE)
_SENTENCE_START_RE = re.compile(r"(?:^|[.!?\n]\s*)$")

_NEWLINE_RE = re.compile(
    r"[ \t]*\b(new\s?line|new\s?paragraph)\b[,.;:!?]?[ \t]*", re.IGNORECASE
)


def remove_fillers(text: str) -> str:
    def repl(m: re.Match) -> str:
        before = text[: m.start()]
        return _CAP if _SENTENCE_START_RE.search(before) else ""

    return _FILLER_RE.sub(repl, text)


def apply_voice_commands(text: str) -> str:
    def repl(m: re.Match) -> str:
        nl = "\n\n" if "para" in m.group(1).lower() else "\n"
        return nl + _CAP

    return _NEWLINE_RE.sub(repl, text)


_SCRATCH_RE = re.compile(r"\b(?:scratch that|delete that|strike that)\b[.,!?]*\s*", re.IGNORECASE)


def apply_scratch_that(text: str) -> str:
    """'Send it Friday. Scratch that. Send it Monday.' -> 'Send it Monday.'"""
    while (m := _SCRATCH_RE.search(text)):
        head = text[: m.start()].rstrip()
        # Drop the sentence that came right before the command.
        body = head.rstrip(".!?,;")
        cut = max(body.rfind(c) for c in ".!?\n")
        text = (head[: cut + 1] + " " if cut >= 0 else "") + _CAP + text[m.end():]
    return text


def apply_terms(text: str, terms: list[str], threshold: float = 0.84) -> str:
    """Snap near-misses to your dictionary spelling ("Kubernetis" -> "Kubernetes")."""
    import difflib

    for term in sorted(terms, key=len, reverse=True):
        n = len(term.split())
        key = re.sub(r"\W", "", term.lower())
        if len(key) < 4:
            continue
        words = list(re.finditer(r"[\w'’-]+", text))
        for size in {max(1, n - 1), n, n + 1}:
            for i in range(len(words) - size, -1, -1):
                span = words[i:i + size]
                cand = text[span[0].start():span[-1].end()]
                ckey = re.sub(r"\W", "", cand.lower())
                if cand == term or not ckey or ckey[0] != key[0]:
                    continue
                if difflib.SequenceMatcher(None, ckey, key).ratio() >= threshold:
                    text = text[:span[0].start()] + term + text[span[-1].end():]
                    words = list(re.finditer(r"[\w'’-]+", text))
                    break
    return text


def apply_replacements(text: str, replacements: dict[str, str]) -> str:
    # Longest phrases first so "open ai api" wins over "open ai".
    for src in sorted(replacements, key=len, reverse=True):
        dst = replacements[src]
        if not src:
            continue
        pattern = rf"(?<!\w){re.escape(src)}(?!\w)"
        text = re.sub(pattern, lambda _m, d=dst: d, text, flags=re.IGNORECASE)
    return text


def tidy(text: str) -> str:
    text = re.sub(r"[ \t]+([,.!?;:])", r"\1", text)  # space before punctuation
    text = re.sub(r",+(?=[.!?])", "", text)  # ",." -> "."
    text = re.sub(r",{2,}", ",", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)
    # Leading punctuation left behind after removing a filler.
    text = re.sub(rf"^[\s,;:.{_CAP}]*", lambda m: _CAP if _CAP in m.group() else "", text)
    text = re.sub(rf"{_CAP}([^\w]*)(\w)", lambda m: m.group(1) + m.group(2).upper(), text)
    return text.replace(_CAP, "").strip(" \t")


def process(text: str, *, fillers: bool = True, commands: bool = True,
            replacements: dict[str, str] | None = None, terms: list[str] | None = None) -> str:
    text = text.strip()
    if fillers:
        text = remove_fillers(text)
    if commands:
        text = apply_scratch_that(text)
        text = apply_voice_commands(text)
    if replacements:
        text = apply_replacements(text, replacements)
    if terms:
        text = apply_terms(text, terms)
    return tidy(text)
