"""Per-app writing styles: casual in chats, proper in email, your exact words in code.

``category(app, title, is_terminal)`` sorts the focused app into chat / email / code /
other. ``apply(text, style)`` adjusts the finished text; the "exact" style also tells the
app to skip the AI rewrite, so prompts and commands are never reworded.
"""

from __future__ import annotations

import re

CATEGORIES = ("chat", "email", "code", "other")
STYLES = ("formal", "casual", "very_casual", "exact")
DEFAULTS = {"chat": "casual", "email": "formal", "code": "exact", "other": "formal"}

CATEGORY_LABELS = {
    "chat": ("Chats", "Slack, Teams, WhatsApp, Discord, Messages…"),
    "email": ("Email", "Outlook, Mail, Gmail, Thunderbird…"),
    "code": ("Code & terminals", "VS Code, Cursor, JetBrains, Terminal, Claude Code…"),
    "other": ("Everything else", "Docs, notes, browsers, AI chats…"),
}
STYLE_LABELS = {
    "formal": "Formal · proper sentences",
    "casual": "Casual · no full stop at the end",
    "very_casual": "Very casual · lowercase, no full stop",
    "exact": "Exact · your words, no AI rewrite",
}

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
