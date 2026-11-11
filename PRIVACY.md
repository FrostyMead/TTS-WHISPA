# Privacy

**Your voice stays on your computer.** Murmr has no servers and no accounts.

- Speech recognition, the AI clean-up and read-aloud run on your computer.
- Murmr never uploads your recordings, transcripts, dictionary or voice models, and nothing
  is sent to the developer. The only network traffic is downloading the app, its updates and
  the speech / AI / voice models (public files from GitHub and Hugging Face).

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
