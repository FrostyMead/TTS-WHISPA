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


def apply_terms(text: str, terms: list[str], threshold: float = 0.84, min_len: int = 4,
                case_only: bool = True) -> str:
    """Snap near-misses to your dictionary spelling ("Kubernetis" -> "Kubernetes").

    ``case_only=False`` is the careful mode for names seen on screen: only words the
    speech model already capitalised as a name are changed, never just their case
    ("Will" on screen doesn't touch "will"), and never a mere suffix ("Gather"/"gathers").
    """
    import difflib

    for term in sorted(terms, key=len, reverse=True):
        n = len(term.split())
        key = re.sub(r"\W", "", term.lower())
        if len(key) < min_len:
            continue
        words = list(re.finditer(r"[\w'’-]+", text))
        for size in {max(1, n - 1), n, n + 1}:
            for i in range(len(words) - size, -1, -1):
                span = words[i:i + size]
                cand = text[span[0].start():span[-1].end()]
                ckey = re.sub(r"\W", "", cand.lower())
                if cand == term or not ckey or ckey[0] != key[0]:
                    continue
                if not case_only and (ckey == key or not cand[:1].isupper()
                                      or ckey.startswith(key) or key.startswith(ckey)):
                    continue  # same word, not heard as a name, or just "Gather"/"gathers"
                if difflib.SequenceMatcher(None, ckey, key).ratio() >= threshold:
                    text = text[:span[0].start()] + term + text[span[-1].end():]
                    words = list(re.finditer(r"[\w'’-]+", text))
                    break
    return text


_NAME_STOP = {
    "i", "i'm", "i'll", "i've", "i'd", "the", "a", "an", "and", "or", "but", "if", "in", "on",
    "at", "to", "for", "of", "it", "is", "this", "that", "we", "you", "he", "she", "they", "my",
    "your", "our", "new", "file", "edit", "view", "help", "home", "search", "settings", "share",
    "save", "open", "close", "reply", "forward", "send", "delete", "copy", "paste", "insert",
    "format", "tools", "window", "draft", "inbox", "sent", "today", "yesterday", "tomorrow",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "january",
    "february", "march", "april", "may", "june", "july", "august", "september", "october",
    "november", "december", "ok", "yes", "no", "hi", "hello", "thanks", "dear", "regards",
}
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9'’_-]*[A-Za-z0-9]")


def candidate_names(text: str, limit: int = 80) -> list[str]:
    """Words in on-screen text that look like names: capitalised mid-sentence (Siobhan,
    Kubernetes), camelCase or with digits (useEffect, GPT4o). Most frequent first."""
    from collections import Counter

    counts: Counter[str] = Counter()
    for line in text.splitlines():
        for sentence in re.split(r"(?<=[.!?:])\s+", line):
            for i, m in enumerate(_TOKEN_RE.finditer(sentence)):
                w = m.group().strip("'’-_")
                if len(w) < 4 or w.lower() in _NAME_STOP:
                    continue
                inner_caps = any(c.isupper() for c in w[1:]) and any(c.islower() for c in w)
                has_digit = any(c.isdigit() for c in w) and any(c.isalpha() for c in w)
                named = w[0].isupper() and w[1:].islower() and i > 0
                if inner_caps or has_digit or named:
                    counts[w] += 1
    return [w for w, _ in counts.most_common(limit)]


def _snippet_pattern(trigger: str) -> re.Pattern:
    words = re.findall(r"[\w']+", trigger.lower())
    # Spoken triggers come back with any punctuation/spacing: "my email", "My email.", "my, email"
    body = r"[\s,.\-]*".join(re.escape(w) for w in words)
    return re.compile(rf"(?<![\w']){body}(?![\w'])", re.IGNORECASE)


def find_snippets(text: str, snippets: dict[str, str]) -> list[str]:
    """Triggers that appear in the text (longest first, so "my work email" beats "my email")."""
    found = []
    for trigger in sorted(snippets, key=len, reverse=True):
        if trigger.strip() and _snippet_pattern(trigger).search(text):
            found.append(trigger)
    return found


def expand_snippets(text: str, snippets: dict[str, str]) -> str:
    """Replace spoken triggers with their saved text. A dictation that is only a trigger
    ("My email.") becomes exactly the snippet, without the transcript's punctuation."""
    for trigger in find_snippets(text, snippets):
        pat = _snippet_pattern(trigger)
        expansion = snippets[trigger]
        if re.fullmatch(r"[\s,.!?;:]*", pat.sub("", text)):
            return expansion
        text = pat.sub(lambda _m, e=expansion: e, text)
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
