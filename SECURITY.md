# Security

Murmr runs entirely on your computer: audio, transcripts and voice samples never leave it.

Updates:
- Only from this repository, and only a commit whose CI run passed. If that can't be
  confirmed (GitHub unreachable, no passing run), nothing is installed.
- Every downloaded file is checked against the commit's own git hashes, with size limits,
  before anything is unpacked into place.
- A version whose packages changed gets its own package environment, so going back to the
  previous version also goes back to its packages.
- What this doesn't cover: someone with write access to this repository (or GitHub itself)
  could still publish a malicious commit that passes CI. Signed releases would close that gap.

The one-click local AI setup only installs Ollama's official release when its SHA-256
checksum is published and matches.

Sync between your computers listens on your local network only while it's switched on,
accepts a handful of connections at a time, and gives nothing to a computer that hasn't
proved it was paired.

Found a security problem? Please report it privately:
**Security → Report a vulnerability** on this repository (don't open a public issue).
