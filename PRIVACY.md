# Privacy

**Your voice stays on your computer.** Murmr has no servers and no accounts.

- Speech recognition, the AI clean-up and read-aloud run on your computer.
- Murmr never uploads your recordings, transcripts, dictionary or voice models, and nothing
  is sent to the developer.
- **The AI clean-up** talks to a server on this computer (Ollama). You can point it at
  another server in config.json, but Murmr refuses to send text anywhere that isn't this
  computer unless you also set `"allow_remote": true`. Then the text you dictate (and
  meeting transcripts you summarize) goes to that server, under its own privacy terms.
- **History:** your dictations are kept on this computer for a year by default (Settings →
  History: forever, 1 year, 90/30/7 days, or off) so you can find and paste them again.
  "Delete all history" removes them now. The data folder is readable only by your user
  account, and an AI API key, if you use one, is kept in the system keychain / Credential
  Manager rather than in a file. The only network traffic is downloading the app, its updates and
  the speech / AI / voice models (public files from GitHub and Hugging Face).
- **Sync (off by default)** only talks to your own computers that you paired with a code, on
  your local network, encrypted. It shares your Dictionary and, if you choose, speech models
  and voices made from your recordings. Never recordings, history or meetings.
- **Report a problem** saves a file on your computer; nothing is sent unless you send it. It
  contains logs and settings with learned words, meeting titles and your user name removed.
- **Meeting notes with computer audio on macOS** need Screen & System Audio Recording
  permission; only the sound is used, never the screen, and no audio is kept.

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
model and voice from your computer in one go. Because nothing was ever uploaded, nothing is
left anywhere else.

Where it's stored: `%APPDATA%\Murmr` (Windows), `~/Library/Application Support/Murmr`
(macOS), `~/.config/murmr` (Linux). Installs from before the rename use the `Whispa` folder.
