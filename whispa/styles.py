"""Dictation modes: how your words come out.

A mode is picked per kind of app (chat / email / code / everything else), or chosen by hand
from the tray, Settings or a shortcut (``Config.mode``; "auto" = per app):

* exact      your words as heard (fillers out, corrections applied), never an AI rewrite
* casual     Message: light tidy, natural tone, no full stop on a one-line message
* very_casual  lowercase, no full stop
* formal     proper sentences
* email      greeting, paragraphs and sign-off on their own lines
* notes      your points as a tidy list, nothing invented
* technical  identifiers, file names, commands and code kept exactly as said
* custom     your own instructions (Settings > Dictation modes > Custom instructions)

``category(app, title, is_terminal)`` sorts the focused app; ``apply(text, style)`` adjusts
the finished text; ``prompt_for(style)`` is what the AI clean-up is told for that mode.
"""

from __future__ import annotations

import re

CATEGORIES = ("chat", "email", "code", "other")
STYLES = ("formal", "casual", "very_casual", "exact", "email", "notes", "technical", "custom")
# Modes the AI always runs for (they reshape the text), and how strictly its answer is checked.
AI_ALWAYS = {"email", "notes", "technical", "custom"}
MODE_CYCLE = ("auto", "casual", "formal", "email", "notes", "technical", "exact", "custom")
DEFAULTS = {"chat": "casual", "email": "formal", "code": "exact", "other": "formal"}

CATEGORY_LABELS = {
    "chat": ("Chats", "Slack, Teams, WhatsApp, Discord, Messages…"),
    "email": ("Email", "Outlook, Mail, Gmail, Thunderbird…"),
    "code": ("Code & terminals", "VS Code, Cursor, JetBrains, Terminal, Claude Code…"),
    "other": ("Everything else", "Docs, notes, browsers, AI chats…"),
}
STYLE_LABELS = {
    "formal": "Formal · proper sentences",
    "casual": "Message · light tidy, no full stop at the end",
    "very_casual": "Very casual · lowercase, no full stop",
    "exact": "Exact · your words, no AI rewrite",
    "email": "Email · greeting, paragraphs, sign-off",
    "notes": "Notes · your points as a tidy list",
    "technical": "Technical · code and names kept exactly",
    "custom": "Custom · your own instructions",
}
MODE_NAMES = {"auto": "Automatic", "formal": "Formal", "casual": "Message", "very_casual": "Very casual",
              "exact": "Exact", "email": "Email", "notes": "Notes", "technical": "Technical",
              "custom": "Custom"}

_PROMPTS = {
    "email": "Format it as an email. Keep every sentence the speaker said, in order, with proper "
             "punctuation and capitalisation. If they start with a greeting (\"Hi Sam\"), put it "
             "on its own line followed by a blank line. If they end with a sign-off (\"Thanks, "
             "Angus\"), put it after a blank line, with the name on its own line. Split a long "
             "body into short paragraphs. Don't add a greeting, sign-off, subject or anything "
             "they didn't say.",
    "notes": "Format it as notes: one point per line, each line starting with \"- \", in the "
             "speaker's order. Write each point as a clean phrase with proper capitalisation "
             "(names capitalised). Drop filler like \"okay so\"; never add facts, conclusions "
             "or points they didn't say.",
    "technical": "This is technical text. Keep code identifiers, file names, paths, commands, "
                 "flags, URLs, version numbers and product names exactly as said, including "
                 "their case (camelCase, snake_case, kebab-case). Don't add punctuation inside "
                 "them and don't expand abbreviations.",
}


def prompt_for(style: str, custom: str = "") -> str:
    """Extra instructions for the AI clean-up in this mode ("" = the normal clean-up)."""
    if style == "custom":
        return custom.strip()
    return _PROMPTS.get(style, "")


def mode_name(mode: str) -> str:
    return MODE_NAMES.get(mode, mode)

# Lower-case process names (Windows, without .exe) and app names (macOS).
_APPS = {
    "chat": {"slack", "discord", "teams", "ms-teams", "msteams", "whatsapp", "whatsapp.root",
             "telegram", "signal", "messages", "messenger", "wechat", "weixin", "line", "skype",
             "element", "beeper", "zulip", "mattermost", "rocket.chat", "viber", "facebook messenger"},
    "email": {"outlook", "olk", "hxoutlook", "mail", "thunderbird", "spark", "spark desktop",
              "airmail", "mimestream", "superhuman", "hey", "mailbird", "em client", "mailspring",
              "canary mail", "proton mail", "postbox"},
    "code": {"code", "code - insiders", "visual studio code", "cursor", "windsurf", "zed", "kiro",
             "sublime_text", "sublime text", "idea64", "idea", "intellij idea", "pycharm64",
             "pycharm", "webstorm64", "webstorm", "rider64", "rider", "clion64", "clion",
             "goland64", "goland", "phpstorm64", "rubymine64", "datagrip64", "studio64",
             "android studio", "xcode", "devenv", "notepad++", "vim", "gvim", "nvim", "neovide",
             "emacs", "windowsterminal", "wt", "cmd", "powershell", "pwsh", "conhost", "terminal",
             "iterm2", "iterm", "warp", "alacritty", "kitty", "wezterm-gui", "wezterm", "hyper",
             "ghostty", "mintty", "tabby", "putty", "mobaxterm", "termius", "nova", "fleet",
             "positron", "trae", "void"},
}
# Web apps in a browser, recognised by the window title.
_TITLES = {
    "email": re.compile(r"\b(gmail|outlook|inbox|proton mail|fastmail|yahoo mail|icloud mail)\b", re.I),
    "chat": re.compile(r"\b(slack|discord|whatsapp|messenger|microsoft teams|telegram|google chat)\b", re.I),
    "code": re.compile(r"\b(github\.dev|vscode\.dev|codespaces|stackblitz|codesandbox|replit)\b", re.I),
}
_BROWSERS = {"chrome", "msedge", "firefox", "brave", "opera", "vivaldi", "arc", "safari",
             "google chrome", "microsoft edge", "zen", "librewolf", "chromium", "thorium"}


def category(app: str, title: str = "", is_terminal: bool = False) -> str:
    if is_terminal:
        return "code"
    name = (app or "").strip().lower()
    for cat, names in _APPS.items():
        if name in names:
            return cat
    if title and (name in _BROWSERS or not name):
        for cat, pattern in _TITLES.items():
            if pattern.search(title):
                return cat
    return "other"


def style_for(styles: dict[str, str], app: str, title: str = "", is_terminal: bool = False) -> str:
    cat = category(app, title, is_terminal)
    style = styles.get(cat, DEFAULTS[cat])
    return style if style in STYLES else DEFAULTS[cat]


_SENTENCE_START = re.compile(r"(^|[.!?]\s+|\n\s*)([A-Z][\w'’]*)")


def apply(text: str, style: str, keep: set[str] | frozenset[str] = frozenset()) -> str:
    """Restyle finished text. ``keep``: words that stay capitalised (names you use)."""
    if style in ("casual", "very_casual"):
        # Chats: a lone message doesn't end with a full stop ("..." and "?" stay).
        if "\n" not in text.strip() and re.search(r"(?<!\.)\.$", text.strip()):
            text = text.strip()[:-1]
    if style == "very_casual":
        keep_l = {k.lower() for k in keep}

        def lower(m: re.Match) -> str:
            word = m.group(2)
            if (word == "I" or word.startswith(("I'", "I’")) or word.isupper() and len(word) > 1
                    or word.lower() in keep_l or any(c.isupper() for c in word[1:])):
                return m.group(0)
            return m.group(1) + word[0].lower() + word[1:]

        text = _SENTENCE_START.sub(lower, text)
    return text
