# A read-aloud voice that sounds like you

These scripts train a [Piper](https://github.com/OHF-Voice/piper1-gpl) voice on the sentences
you read on Murmr's **Voice training** page, and install it into Murmr, where it shows up in
**Settings → Read aloud → Voice** as "My voice" (Windows and Mac: Piper voices are small ONNX
files that run fast on any CPU).

Everything runs on your own computer. Your recordings never leave it.

## What's used

- Only with your OK: **Settings → Your voice → Create a voice that sounds like me**.
- Only sentences you **read live** on the Voice training page, never dictations or other audio.
- Only clean takes: the high-quality (24 kHz) copy Murmr keeps of each read, skipping noisy,
  too-quiet or clipped ones. A quiet room and your usual mic give the best voice.
- **How much:** 15 minutes gives a rough voice, 30-60 minutes starts to sound natural, more
  keeps improving it.

## Requirements

Linux or WSL2 with an NVIDIA GPU (8 GB is enough at batch size 16), a C compiler (`gcc`),
about 4 GB of disk for the tools and 1-2 GB per voice.

## Setup (once)

```bash
bash setup.sh            # ~/murmr-voice: uv, Python 3.11, PyTorch (CUDA), Piper training
```

## Train

From this folder, with `~/murmr-voice/.venv` active:

```bash
python prepare_voice.py               # your clean reads -> ~/murmr-voice/work/me/dataset
python train_voice.py --hours 6       # one round; run it again (e.g. each night) to continue
python export_voice.py                # export, check it's understandable, install into Murmr
```

`./pipeline.sh` runs all three. Close Murmr and the local AI first (they use GPU memory).

- **Rounds:** the first round starts from a finished public voice (Lessac, medium quality),
  which is what makes a voice from an hour of recordings possible. Every later round resumes
  where the last stopped. Piper's guidance for under 5 hours of audio is about 2000 epochs;
  on an RTX 3070 Ti with an hour of recordings that's roughly 15-25 hours in total, so a few
  nights. The voice gets better each round, and you can install it after any round.
- **Accent:** `--espeak-voice en-gb-x-rp` on the first round for British, Australian or NZ
  English (it's the pronunciation guide the voice learns from; the default is `en-us`).
- **The check:** `export_voice.py` reads four sentences with the new voice and has Parakeet
  transcribe them. It only installs the voice if fewer than 1 word in 4 is misheard, and saves
  `sample.wav` so you can listen first.

## License note

Piper's training code is GPL-3.0, and the starting voice (Lessac) comes from the
[rhasspy/piper-checkpoints](https://huggingface.co/datasets/rhasspy/piper-checkpoints)
collection. That's fine for your own voice on your own computer; check both before
distributing voices commercially.
