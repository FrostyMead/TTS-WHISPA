# Fine-tune Parakeet on your voice

These scripts take the clips you recorded in Whispa (**Voice training** page, plus dictations you
corrected), fine-tune NVIDIA Parakeet TDT 0.6B v3 on them with NeMo, and export the result to the
ONNX layout Whispa runs. The tuned model is then compared with the base model on held-out clips
of your voice and on general speech. It's only recommended if it's better for you and no worse
for everyone else. You switch to it, and back, in Whispa under **Settings → Speech model**.

Everything runs locally. Your clips never leave the machine and are never committed.

## Requirements

| | |
|---|---|
| OS | Linux, or **WSL2** on Windows (NeMo doesn't support native Windows) |
| GPU | NVIDIA, 8 GB VRAM (tested on an RTX 3070 Ti). The WSL2 driver comes from the Windows NVIDIA driver; you don't need a CUDA toolkit |
| Disk | ~8 GB for the Python environment, ~2.5 GB for the base model, ~0.7 GB for general speech data, ~3.5 GB per exported model |
| RAM | 16 GB (WSL2's default share of a 32 GB PC is enough) |
| Data | **30+ minutes** of `read`/`corrected` clips for a first useful tune; 60+ is better |

**Free the GPU before training.** Quit Whispa (tray → Quit) and unload the local AI
(`ollama stop gemma3:4b`). Together they hold 3–5 GB of VRAM. Restart Whispa when the run is done.

## One-time setup (WSL2 Ubuntu)

Everything goes in `~/whispa-finetune`, with no system packages and no sudo:

```bash
R=~/whispa-finetune; mkdir -p $R/bin && cd $R
curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=$R/bin UV_NO_MODIFY_PATH=1 sh
bin/uv venv --python 3.12 .venv && . .venv/bin/activate
bin/uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
bin/uv pip install "nemo_toolkit[asr]" onnx onnxruntime onnx-asr soundfile
python -c "import torch; print(torch.cuda.is_available())"     # must print True
```
This downloads about 4 GB and takes ~10 minutes. Tested with torch 2.11 (cu128), NeMo 3.0.0,
Lightning 2.4 and Python 3.12. The first `import nemo` takes a few minutes; after that it's quick.

## Run it

From the repo's `scripts/finetune` folder inside WSL (e.g. `/mnt/c/Users/<you>/TTS-WHISPA/scripts/finetune`),
with the venv active:

```bash
./pipeline.sh                  # all steps below, with the defaults
```
or step by step:

| Step | Command | What it does |
|---|---|---|
| 1 | `python prepare_data.py` | Reads `%APPDATA%\Whispa\training` (found automatically from WSL). Copies `read` + `corrected` clips into `~/whispa-finetune/work/data/user` and splits them **train / val / test (75 / 10 / 15%)** by a hash of each clip id, so a test clip stays a test clip forever. `unknown` clips are never used; `--include-unedited` adds dictations you left as they were. |
| 2 | `python get_general_data.py` | One-time download, about 700 MB: LibriSpeech dev-clean (replay data) and test-clean (benchmark), with **LibriSpeech-PC** punctuated, capitalized transcripts. |
| 3 | `python train.py` | Fine-tunes and writes `work/models/tuned-<date>/model.nemo` and `train_info.json`. |
| 4 | `python export_onnx.py work/models/tuned-<date>` | Exports `onnx/` in the onnx-asr layout (config.json, encoder-model.onnx + .data, decoder_joint-model.onnx, vocab.txt, int8 copies for CPU). |
| 5 | `python evaluate.py work/models/tuned-<date>` | Runs base and tuned models through **onnx-asr** (what Whispa runs) on your held-out clips and 200 general utterances. Prints WER, PnC-WER and changed transcripts, and saves `onnx/eval.json`. |
| 6 | `python install_model.py work/models/tuned-<date>` | Copies the model to `%APPDATA%\Whispa\models\` (refuses if step 5 didn't recommend it; `--force` overrides). Then in Whispa: **Settings → Speech model → Use**. |

### The training recipe (train.py)

Full fine-tuning of 0.6B parameters with AdamW needs roughly 10 GB plus activations, which is
more than 8 GB. More importantly, with half an hour of one voice it would overfit and forget
general speech. So:

- **Partial freezing.** Only the top `--train-layers 6` of the 24 FastConformer encoder layers
  train, plus the TDT decoder and joint network (about 150M parameters). The lower layers
  (general acoustics) stay fixed, and so do **all BatchNorm statistics**, which a small,
  single-speaker set would otherwise pull towards your mic and room. This also exports as
  plain weights. Adapters or LoRA would mean extra modules that must be merged before export.
- **Replay against forgetting.** Every epoch mixes in LibriSpeech-PC utterances, `--replay 1.0`×
  as much audio as your own clips. Because those transcripts have punctuation and capitals, the
  model keeps writing them.
- **Gentle optimization.** AdamW at lr 2e-5 (cosine to 2e-6, 10% warm-up), weight decay 1e-3,
  gradient clipping 1.0, bf16 mixed precision, batch 4 × accumulation 4, SpecAugment from the
  base config. The default is 8 epochs over the mix.
- **Checkpoint choice.** The best checkpoint is picked by WER on **your validation split**. The
  test split is only ever read by `evaluate.py`.

Knobs worth trying when the first result isn't good enough: `--train-layers 8` or `12` (more
capacity; watch VRAM), `--epochs 4` or `12`, `--lr 1e-5` or `5e-5`, `--replay 0.5` or `2`. Keep
whichever evaluates best: the held-out test set makes the comparison fair.

### When does it recommend switching?

`evaluate.py` says **YES** only if both of these hold:
- your WER improves by at least 5% relative (e.g. 8.0% → 7.6% or better), and
- general WER gets no worse than +0.5 points absolute or +5% relative, whichever is larger.

WER here ignores case, punctuation and number formatting, like the Open ASR Leaderboard.
PnC-WER, which counts punctuation and capitals too, is printed alongside.

## Time and memory

MEASURED_PLACEHOLDER

## Privacy

- Prepared data, checkpoints and exports live in `~/whispa-finetune/work` inside WSL. Delete that
  folder to remove them; your originals stay in `%APPDATA%\Whispa\training`.
- Nothing is uploaded. The only network access is the one-time downloads: Python packages, the
  base model from Hugging Face (`nvidia/parakeet-tdt-0.6b-v3`) and LibriSpeech from openslr.org.
- The repo's `.gitignore` doesn't need entries for any of this: the scripts never write into the
  repo.
