# Privacy

**Your voice is turned into text on your computer.** Murmr has no servers and no accounts,
and the developer never receives anything unless you choose to send a problem report. The
exceptions, each one optional and described below: sync to your own other computers, an AI
server you point Murmr at yourself, and problem reports you decide to send.

## What leaves your computer

- **Downloads:** the app and its updates (GitHub), the speech, read-aloud, speaker and local
  AI models (GitHub, Hugging Face, Ollama). These requests send nothing about you or your
  dictation, but like any download they show your IP address to the site.
- **Opening things you ask for:** a website you open with a computer command, or a GitHub
  issue page from Report a problem (with the description you typed).
- **The AI clean-up** talks to a server on this computer (Ollama, or Apple's on-device model on
  a Mac). You can point it at another server in config.json, but Murmr refuses to send it
  anything unless you also set `"allow_remote": true` (only a loopback address or `localhost`
  counts as this computer; requests ignore proxies and redirects; one-click AI setup only
  installs on this computer). With that set, the text you dictate, and meeting transcripts
  you summarize, go to that server under its own privacy terms.
- **Sync (off by default)** talks only to your own computers that you paired with a code, on
  your local network, encrypted. It shares your Dictionary and, only where you switch them on
  (on both computers), speech models and voices made from your recordings. Never recordings,
  history or meetings. Computers that aren't paired get no answer from it (except while
  you're showing a pairing code, when they can see this computer's name).
- **Report a problem** saves a zip on your computer and shows you exactly what's in it;
  nothing is sent unless you send it. See below for what it contains.

## What's kept on your computer

- **History:** your dictations, as heard and as pasted, for a year by default (Settings →
  History: forever, 1 year, 90/30/7 days, or off). "Delete all history" removes them now.
- **Dictionary:** your words, corrections and snippets.
- **Meetings:** transcripts and summaries (text only, no audio) until you delete them.
- **Logs** (`whispa.log`, up to 3 MB, older lines rotate out; `crash.log` if Murmr ever
  freezes): times, which app you dictated into, how long things took and errors. Not what
  you say or write: dictated text, Command Mode instructions, AI answers, learned words,
  snippet and meeting text, and your settings' text values are kept out of them.
- **Clipboard:** to paste, Murmr puts your text on the clipboard for a moment, then puts
  back what was there (text only). It marks the text so Windows' Clipboard history and cloud
  clipboard, and Mac clipboard managers that respect the standard markers, don't keep it. To
  avoid the clipboard altogether, set `"paste_method": "type"` in config.json.
- **Screen names** (on by default): when you start talking, Murmr reads the names in the
  window you're dictating into, so they're spelled the same. Read on this computer, used for
  that dictation, never saved.
- **The data folder** is `%APPDATA%\Murmr` (Windows), `~/Library/Application Support/Murmr`
  (macOS) or `~/.config/murmr` (Linux); installs from before the rename use `Whispa`. On macOS
  and Linux Murmr makes it readable only by your user account. On Windows it relies on that
  folder's normal per-user permissions (administrators, backups and device management can
  still reach it). An AI API key, if you use one, is kept in the system keychain / Credential
  Manager rather than in a file (on a system without one, it stays in config.json).

## Other people

- **Meeting notes** transcribe everyone in the call or room. Before your first meeting,
  Murmr asks you to confirm you'll let everyone know: in some places it needs everyone's
  consent.
- **Telling voices apart** uses voiceprints (how a voice sounds) only during the meeting; they
  aren't saved, unless:
- **Remembering voices across meetings (off by default).** When it's on and you name someone,
  Murmr asks whether that person agreed before keeping their voiceprint (256 numbers, no audio,
  nothing they said) in `people.json`, so later meetings label them by name. It's never synced
  or uploaded. A voiceprint used to recognise someone is sensitive (biometric) information under
  Australian privacy law and similar laws elsewhere: only remember people who have agreed.
  Forget anyone, or everyone, on the Meetings page.
- **Computer audio in meetings on macOS** needs Screen & System Audio Recording permission;
  only the sound is used, never the screen, and no audio is kept.

## Your voice data (opt-in)

Off until you switch it on, in **Settings → Your voice**:

- **Improve recognition with my voice:** keeps your dictations' audio, and your Voice
  training reading, so the speech model can be tuned to your voice. Tuning runs on your own
  computer.
- **Create a voice that sounds like me:** uses only sentences you read live on the Voice
  training page, never other recordings, to build a read-aloud voice like yours.

Recordings are never kept from password fields or from the apps you list (password
managers by default). On a work-managed computer Murmr warns you that recordings could
contain confidential work information.

**Delete everything about my voice** (Settings → Your voice) removes every recording, tuned
model and voice from this computer. Files still in use (a loaded speech model) are removed the
next time Murmr starts, and Murmr tells you if that's the case. It can't reach copies you
made yourself (exports, backups) or ones you synced to another computer: delete those there.

## Problem reports

The zip has what you wrote, Murmr's version and your OS, your settings (without your
replacements, Custom mode instructions, computer name, AI key or server address) and the
logs (which keep out what you say and write; your user name and home folder are also taken
out). Never audio, recordings, history, your Dictionary or meetings. Murmr shows you the
whole contents before you send it anywhere.
