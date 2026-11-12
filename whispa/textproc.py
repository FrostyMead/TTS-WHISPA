"""Deterministic clean-up applied to raw transcripts."""

from __future__ import annotations

import functools
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


_BULLET_RE = re.compile(r"[ \t]*[,;]?[ \t]*\b(?:new |next )?bullet[ -]?points?\b[,.;:]?[ \t]*",
                        re.IGNORECASE)


def apply_voice_commands(text: str) -> str:
    def repl(m: re.Match) -> str:
        nl = "\n\n" if "para" in m.group(1).lower() else "\n"
        return nl + _CAP

    text = _NEWLINE_RE.sub(repl, text)
    # "Groceries: bullet point milk, bullet point eggs." -> list items on their own lines
    return _BULLET_RE.sub("\n- " + _CAP, text)


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

    prepared = []
    for term in terms:
        key = _KEY_RE.sub("", term.lower())
        if len(key) >= min_len:
            prepared.append((term, key, len(term.split())))
    if not prepared or not text:
        return text
    prepared.sort(key=lambda t: len(t[0]), reverse=True)
    sizes = sorted({s for _, _, n in prepared for s in (max(1, n - 1), n, n + 1)})
    index = _span_index(text, sizes)
    for term, key, n in prepared:
        for size in (max(1, n - 1), n, n + 1) if n > 1 else (1, 2):
            changed = False
            guard = len(text) + 1  # a replaced span: nothing overlapping it is touched
            for start, end, cand, ckey in index.get((key[0], size), ()):
                if cand == term or end > guard:
                    continue
                total = len(ckey) + len(key)
                if 2 * min(len(ckey), len(key)) < threshold * total:
                    continue  # lengths alone rule it out (cheap, and most candidates stop here)
                if not case_only and (ckey == key or not cand[:1].isupper()
                                      or ckey.startswith(key) or key.startswith(ckey)):
                    continue  # same word, not heard as a name, or just "Gather"/"gathers"
                m = difflib.SequenceMatcher(None, ckey, key)
                if m.real_quick_ratio() >= threshold and m.quick_ratio() >= threshold and \
                        m.ratio() >= threshold:
                    text = text[:start] + term + text[end:]  # spans run from the end: safe
                    changed = True
                    guard = start
            if changed:  # every occurrence of this term is fixed; re-read the text once
                index = _span_index(text, sizes)
    return text


_KEY_RE = re.compile(r"\W")
_WORD_RE = re.compile(r"[\w'’-]+")


def _span_index(text: str, sizes: list[int]) -> dict[tuple[str, int], list]:
    """Runs of 1, 2... words in ``text``, by (first letter, number of words), last first."""
    words = list(_WORD_RE.finditer(text))
    out: dict[tuple[str, int], list] = {}
    for size in sizes:
        for i in range(len(words) - size, -1, -1):
            start, end = words[i].start(), words[i + size - 1].end()
            cand = text[start:end]
            ckey = _KEY_RE.sub("", cand.lower())
            if ckey:
                out.setdefault((ckey[0], size), []).append((start, end, cand, ckey))
    return out  # each list runs from the end of the text to the start


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
    """Your corrections, in one pass however many there are. Longest phrases first, so
    "open ai api" wins over "open ai"."""
    if not replacements or not text:
        return text
    pattern, lookup = _replacement_pattern(tuple(sorted(replacements.items())))
    if pattern is None:
        return text
    return pattern.sub(lambda m: lookup.get(m.group(0).lower(), m.group(0)), text)


@functools.lru_cache(maxsize=8)
def _replacement_pattern(items: tuple[tuple[str, str], ...]):
    lookup = {src.lower(): dst for src, dst in items if src}
    if not lookup:
        return None, {}
    alternatives = "|".join(re.escape(src) for src in sorted(lookup, key=len, reverse=True))
    return re.compile(rf"(?<!\w)(?:{alternatives})(?!\w)", re.IGNORECASE), lookup


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
            replacements: dict[str, str] | None = None, terms: list[str] | None = None,
            smart: bool = False) -> str:
    text = text.strip()
    if fillers:
        text = remove_fillers(text)
    if commands:
        text = apply_scratch_that(text)
        text = apply_voice_commands(text)
    if smart:
        from .formatting import smart_format

        text = smart_format(text)
    if replacements:
        text = apply_replacements(text, replacements)
    if terms:
        text = apply_terms(text, terms)
    return tidy(text)
