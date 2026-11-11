# Whispa

Local push-to-talk dictation for **Windows, macOS and Linux**, in the spirit of Wispr Flow.
Hold a key, talk, let go, and the text appears in whatever app has focus. Speech recognition
runs on your machine with **NVIDIA Parakeet TDT 0.6B v3**; no audio leaves your computer.

The pill at the bottom of your screen while you talk:

![The dictation pill: idle, listening, hands-free, transcribing, messages](docs/pill.png)

The app window (tray icon → Open): your dictations, dictionary and settings.

![Home: stats and dictation history](docs/home.png)

| Settings | Dictionary |
|---|---|
| ![Settings](docs/settings.png) | ![Dictionary](docs/dictionary.png) |

No text field selected? You get a copy card instead:

![Copy card](docs/copy-card.png)

Screenshots use sample data. Light and dark mode follow your OS.

**Status:** used daily on Windows 11. The macOS and Linux code paths are written but untested.

## Why Parakeet (research notes, Nov 2026)

| Option | Notes |
|---|---|
| **Parakeet TDT 0.6B v3** (default) | ~6.3% avg WER on the Open ASR Leaderboard (better than Whisper large-v3 at ~7.4%), punctuation and capitalization built in, 25 European languages, CC-BY-4.0. ONNX builds run anywhere through [onnx-asr](https://github.com/istupakov/onnx-asr). |
| Parakeet TDT 0.6B v2 | English only, slightly lower WER (~6.05%). Set `"model": "nemo-parakeet-tdt-0.6b-v2"`. |
| Nemotron-Labs-Unified-ASR 0.6B / Nemotron 3.5 ASR Streaming | NVIDIA's newest models, built for streaming. NeMo/PyTorch only for now (no ONNX build), so not cross-platform yet. Likely future upgrade for live, as-you-speak text. |
| Whisper large-v3-turbo | Works too (`"model": "onnx-community/whisper-large-v3-turbo"`), slower, more languages. |

Measured on this dev box (RTX 3070 Ti / Ryzen 5 5600X), one 6.8 s sentence:
**~45 ms on CUDA**, **~330 ms on CPU (int8)**. Even CPU-only machines feel instant.

## Install

**Windows** (auto-detects an NVIDIA GPU):
```powershell
powershell -ExecutionPolicy Bypass -File install.ps1        # or: -Runtime cpu | directml
.venv\Scripts\whispa.exe
```

**macOS / Linux**:
```bash
./install.sh          # Linux with NVIDIA picks cuda automatically; macOS uses CPU (fast on Apple Silicon)
.venv/bin/whispa
```

The first run downloads the model (~650 MB) into the Hugging Face cache. The Windows installer
adds **Whispa** to the Start Menu and Desktop; it starts without a console window. To add the
shortcuts by hand: `.venv\Scripts\python -m whispa --install-shortcuts`.

## The app

Whispa lives in the system tray. **Click the tray icon** (or launch Whispa again from the Start
Menu) to open the window; closing the window keeps it running in the tray. Right-click the tray
icon for hands-free dictation, recent transcripts, settings, restart and quit.

- **Home**: your dictations grouped by day (search, copy, delete), plus words dictated, speaking
  speed and time saved compared with typing.
- **Dictionary**: your words and learned corrections. Add or remove entries.
- **Settings** (changes apply immediately): record a new push-to-talk shortcut or hands-free key
  by pressing it, pick a microphone, filler removal / voice commands, what happens with no text
  field, AI clean-up status, learning, voice samples, launch at login, sounds, idle pill, and
  GPU/CPU.

Launch at login starts Whispa quietly in the tray (`--background`).

## Using it

| Action | Default (Windows / Linux) | Default (macOS) |
|---|---|---|
| Push-to-talk | hold **Ctrl+Win** (Ctrl+Super), release to insert | hold **fn** |
| Hands-free | while holding, tap **Space**, or **double-tap** the hotkey. Press the hotkey again to finish | same |
| Cancel | **Esc** | **Esc** |
| Mouse | click the idle pill to start hands-free. In hands-free, ✕ cancels and ■ finishes | same |

Pressing another key within 0.4 s of the hotkey (e.g. Ctrl+Win+D) counts as an OS shortcut
and silently cancels, so your normal shortcuts keep working.

### Where the text goes
- **Cursor in a text field:** the text is pasted at the cursor. Whispa **never presses Enter**:
  you review and send it yourself. Line breaks never come last, and in `type` mode they're sent as
  Shift+Enter, so chat apps don't send early.
- **No text field focused** (desktop, a file list, a web page you haven't clicked into): a card
  pops up with the text, already on your clipboard, with **Copy** and **Dismiss** buttons. It
  hides after 20 s unless you hover over it.
- **Can't tell** (custom-drawn apps, games): it pastes.

Detection uses UI Automation on Windows and the Accessibility API on macOS. Linux always pastes for now.

### Clean-up and fixing your mistakes
- filler words ("um", "uh", "hmm") removed
- say **"new line"** / **"new paragraph"** for line breaks, **"scratch that"** to drop the last sentence
- your learned dictionary and `replacements` are applied
- **local LLM polish** (automatic once Ollama is installed): handles self-corrections
  ("meet Tuesday, no wait, Wednesday" → "meet Wednesday"), stutters, false starts, grammar,
  misheard words and dictated lists. It uses your vocabulary and the app you're typing in as
  context. The prompt tells it to keep your words, never answer questions, and never add content.
  It needs Ollama:
  ```
  winget install Ollama.Ollama      # macOS: brew install ollama
  ollama pull gemma3:4b             # ~3.3 GB; or a smaller model, then set llm_cleanup.model
  ```
  Whispa picks it up within 2 minutes, with no restart needed. If the LLM is slow or gives a strange
  answer, the plain transcript is used.

### Personalization: it learns from your fixes
After pasting, Whispa watches that text field for up to 45 s (until you send, switch fields, or
start a new dictation). If you fix a misheard word, it learns the fix and the pill shows
**"Learned: Sarah"**:
- names and jargon (`Sara → Sarah`, `cooper netties → Kubernetes`, `github → GitHub`) are learned
  after one fix
- ordinary words need the same fix twice, so a one-off rewording isn't treated as a mishearing
- changes of mind (`Tuesday → Wednesday`, `their → there`) and full rewrites are ignored
- password fields are never read

Everything is stored in `dictionary.json` (tray → **Open personal dictionary**). Add your own
`terms` there (names, product words): near-misses snap to your spelling, and the LLM is told to
prefer them. Hand edits are picked up immediately.

**Your voice:** each dictation's audio and final text, including your corrections, is saved to
`training/` (up to `training_max_mb`). That's the dataset needed to fine-tune Parakeet on your
voice and accent with NVIDIA NeMo, which is the real way to "learn how you sound". It needs a few
hours of your speech and is a separate, heavier step (Linux/WSL + GPU). Turn collection off with
`save_training_audio: false`.

Recent transcripts are in the tray menu, and the full history is in `history.jsonl` next to the config.

## Settings

Most options are in the Settings window. Everything is also in a JSON file you can edit
(restart Whispa afterwards). Locations:
Windows `%APPDATA%\Whispa\config.json`, macOS `~/Library/Application Support/Whispa/`, Linux `~/.config/whispa/`.

| Key | Default | |
|---|---|---|
| `hotkey` | `ctrl+cmd` / `fn` | `+`-joined: `ctrl alt shift cmd(win/super) fn`, sides like `right_alt`, `ctrl_r`, keys like `f9` |
| `hands_free_key` | `space` | key that locks hands-free while the hotkey is held (`""` disables) |
| `double_tap_to_lock` | `true` | |
| `model` | `nemo-parakeet-tdt-0.6b-v3` | any [onnx-asr model name](https://istupakov.github.io/onnx-asr/usage/) |
| `device` | `auto` | `auto` · `cuda` · `directml` · `coreml` · `cpu` |
| `quantization` | `auto` | int8 on CPU, full precision on GPU; or `int8` / `none` |
| `input_device` | `null` | mic index or name substring (see `whispa-cli --list-devices`) |
| `keep_mic_open` | `false` | `true` keeps the mic hot and adds 300 ms of pre-roll so the first syllable is never clipped |
| `remove_fillers`, `voice_commands`, `append_space` | `true` | |
| `replacements` | `{}` | case-insensitive whole-phrase replacements |
| `llm_cleanup.enabled` | `auto` | `auto` (on when Ollama + model found) · `true` · `false` |
| `llm_cleanup.provider` / `base_url` / `model` | `ollama` / `http://127.0.0.1:11434` / `gemma3:4b` | `provider: "openai"` for LM Studio, llama.cpp server etc. (`base_url` ending in `/v1`) |
| `llm_cleanup.keep_alive`, `timeout_s`, `prompt` | `30m`, `6`, built-in | |
| `learn_from_edits` | `true` | watch your edits after pasting and learn corrections |
| `save_training_audio`, `training_max_mb` | `true`, `2000` | keep audio + final text pairs for fine-tuning |
| `no_text_field` | `popup` | `popup` (copy card) · `paste` (paste anyway) |
| `paste_method` | `auto` | `clipboard` · `type` (simulated keystrokes) |
| `paste_shortcut` | platform default | e.g. `ctrl+shift+v` for Linux terminals |
| `sounds`, `sound_volume`, `show_idle_pill`, `save_history` | | |

Using LM Studio instead of Ollama:
```json
"llm_cleanup": { "enabled": "auto", "provider": "openai", "base_url": "http://127.0.0.1:1234/v1", "model": "your-model" }
```

## Platform notes

**macOS**
- System Settings → Keyboard → "Press 🌐 key to" → **Do Nothing**, so fn doesn't open the emoji picker.
- Grant **Microphone**, **Input Monitoring** and **Accessibility** to the app you launch Whispa from
  (Terminal, iTerm, or the Python binary). The fn key and paste need the last two.

**Windows**
- The physical Fn key is handled inside the keyboard and is invisible to Windows, so the default is Ctrl+Win (same as Wispr Flow on Windows).
- No console window: `whispa.exe` is a GUI launcher.

**Linux**
- X11 works out of the box (needs `xclip` or `xsel` for the clipboard).
- Wayland: global key hooks via pynput only work for XWayland apps. Text goes in through
  `wtype` or `ydotool` if one is installed. A native evdev hotkey backend is on the roadmap.
- The tray icon on GNOME needs the AppIndicator extension.

## Diagnostics

```
whispa-cli --test-mic          # record 3 s, report the input level
whispa-cli --list-devices
whispa-cli --transcribe x.wav  # run Parakeet on a file (prints raw + cleaned)
```
Logs: `whispa.log` next to the config (tray → Open log).

## Architecture

```
whispa/
  hotkey.py    PTTStateMachine (pure logic, unit-tested) + pynput hooks + macOS Quartz fn tap
  audio.py     sounddevice capture at 16 kHz, live RMS level, mic fallback
  asr.py       onnx-asr Parakeet, provider auto-select (CUDA > DirectML > CPU), VAD chunking for >60 s
  textproc.py  fillers, voice commands, "scratch that", dictionary replacements + fuzzy terms
  llm.py       local LLM polish (Ollama native API or OpenAI-compatible), with sanity guards
  focus.py     is the cursor in a text field? read its text (Windows UIA / macOS AX)
  learn.py     learn corrections from your edits, personal dictionary, training-data store
  inject.py    clipboard paste with restore / typing / wtype (never presses Enter)
  overlay.py   frameless, non-activating, always-on-top Qt pill with animations
  card.py      "no text field" copy card
  window.py    the app window: Home (history + stats), Dictionary, Settings
  widgets.py   switch, keycap shortcut recorder, stat cards;  theme.py: light/dark styles
  icons.py     app + tray icons drawn in code;  shortcuts.py: Start Menu / Desktop entries
launcher.pyw   windowless start on Windows (base pythonw.exe + the venv's packages; some venv
               launcher stubs start console python.exe, which pops up a terminal)
  app.py       wiring, tray menu, history, single-instance lock
```

## Roadmap

- Fine-tuning script: training/ manifest → NeMo fine-tune → ONNX export for onnx-asr
- Streaming partial results with Nemotron streaming models (once ONNX exports exist)
- MLX backend (`parakeet-mlx`) for Apple Silicon GPU
- evdev hotkeys for Wayland, layer-shell overlay
- Per-app styles (casual in Slack, formal in email) through the LLM pass
- Packaged installers (PyInstaller / Briefcase)

## Tests

```
.venv/Scripts/python -m pytest      # Windows
.venv/bin/python -m pytest          # macOS / Linux
```

## License

Whispa is MIT licensed (see `LICENSE`). The Parakeet model it downloads is NVIDIA's, released
under CC-BY-4.0; the ONNX conversion used here comes from
[istupakov/parakeet-tdt-0.6b-v3-onnx](https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx).
