"""Smart formatting: turn spoken forms into written ones, without the AI.

* emails:   "john dot smith at gmail dot com"  -> john.smith@gmail.com
* websites: "example dot com", "whispa dot io slash docs" -> example.com, whispa.io/docs
* money:    "20 dollars", "10 euros and 50 cents", "$20 dollars" -> $20, €10.50, $20
            (not pounds: "lost 5 pounds" is usually weight)
* percent:  "50 percent", "12.5 per cent" -> 50%, 12.5%
* times:    "330 pm", "3.30 p.m.", "3 30 pm" -> 3:30 pm

Every rule needs an unambiguous shape (a real domain ending, a number next to the unit),
so ordinary sentences ("the dot com bubble", "at the station") are left alone.
"""

from __future__ import annotations

import re

_TLDS = ("com|org|net|io|ai|dev|app|co|uk|nz|au|ca|us|de|fr|es|it|nl|ie|in|edu|gov|me|info|"
         "biz|tv|xyz|tech|cloud|so|gg|ly|sh|fm")
_DOT = r"(?:\s+dot\s+|\.)"
_LABEL = r"[a-z0-9][a-z0-9-]*"
# Spoken domains must start with a name, not "the dot com bubble".
_NOT_NAME = r"(?!(?:the|a|an|this|that|my|our|your|his|her|its|their|some|any)\s)"
# A domain ends with a known top-level domain: "gmail dot com", "acme dot co dot uk".
_DOMAIN = rf"{_LABEL}(?:{_DOT}{_LABEL})*?{_DOT}(?:{_TLDS})(?:{_DOT}(?:{_TLDS}))?"
_LOCAL_SEP = r"(?:\s+dot\s+|\s+underscore\s+|\s+dash\s+|[._-])"
_LOCAL = rf"{_LABEL}(?:{_LOCAL_SEP}{_LABEL}){{0,3}}"

_EMAIL_RE = re.compile(rf"(?<![\w@.]){_NOT_NAME}({_LOCAL})\s+at\s+({_DOMAIN})(?![\w-])",
                       re.IGNORECASE)
_URL_RE = re.compile(rf"(?<![\w@./]){_NOT_NAME}((?:www{_DOT})?{_DOMAIN})((?:\s+slash\s+[a-z0-9-]+)*)"
                     rf"(?![\w-])", re.IGNORECASE)


def _join(spoken: str) -> str:
    spoken = re.sub(r"\s+dot\s+", ".", spoken, flags=re.IGNORECASE)
    spoken = re.sub(r"\s+underscore\s+", "_", spoken, flags=re.IGNORECASE)
    spoken = re.sub(r"\s+dash\s+", "-", spoken, flags=re.IGNORECASE)
    return spoken.lower()


def emails(text: str) -> str:
    return _EMAIL_RE.sub(lambda m: f"{_join(m.group(1))}@{_join(m.group(2))}", text)


def websites(text: str) -> str:
    def repl(m: re.Match) -> str:
        if " " not in m.group(1) and not m.group(2):
            return m.group(0)  # already written as a domain
        path = re.sub(r"\s+slash\s+", "/", m.group(2), flags=re.IGNORECASE)
        return _join(m.group(1)) + path.lower()

    return _URL_RE.sub(repl, text)


_NUM = r"\d+(?:[.,]\d+)*"
_CURRENCY = {"dollars": "$", "dollar": "$", "euros": "€", "euro": "€"}
_MONEY_RE = re.compile(rf"(?:([$€])\s?)?\b({_NUM})\s+(dollars?|euros?)\b"
                       rf"(?:\s+(?:and\s+)?(\d{{1,2}})\s+cents?\b)?", re.IGNORECASE)
_PERCENT_RE = re.compile(rf"\b({_NUM})\s*(?:percent|per\s+cent)\b", re.IGNORECASE)
# "330 pm" works too: the hour backtracks to "3" so the minutes can be "30".
_TIME_RE = re.compile(r"\b(\d{1,2})(?:[.:\s]?([0-5]\d))?\s?(a\.?m\.?|p\.?m\.?)(?=\W|$)", re.IGNORECASE)


def money(text: str) -> str:
    def repl(m: re.Match) -> str:
        amount = m.group(2)
        if m.group(4):
            amount += "." + m.group(4).rjust(2, "0")
        return f"{m.group(1) or _CURRENCY[m.group(3).lower()]}{amount}"

    return _MONEY_RE.sub(repl, text)


def percent(text: str) -> str:
    return _PERCENT_RE.sub(r"\1%", text)


def times(text: str) -> str:
    def repl(m: re.Match) -> str:
        hour, minute, ampm = m.group(1), m.group(2), m.group(3)
        if not 1 <= int(hour) <= 12:
            return m.group(0)
        suffix = ampm.lower().replace(".", "")
        # A final dot also ends the sentence: "pm." always; "p.m." at the end of a sentence.
        rest = text[m.end():]
        if ampm.endswith(".") and ("." not in ampm[:-1] or not rest.strip()
                                   or re.match(r"\s*(\n|[A-Z])", rest)):
            suffix += "."
        return f"{int(hour)}:{minute} {suffix}" if minute else f"{int(hour)} {suffix}"

    return _TIME_RE.sub(repl, text)


def smart_format(text: str) -> str:
    text = emails(text)
    text = websites(text)
    text = money(text)
    text = percent(text)
    return times(text)
