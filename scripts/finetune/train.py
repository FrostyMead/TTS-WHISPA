"""Step 3: fine-tune Parakeet TDT 0.6B v3 on your clips (NeMo, Linux/WSL2, NVIDIA GPU).

    python train.py                       # defaults sized for an 8 GB GPU
    python train.py --steps 30 --tiny     # quick pipeline check

Recipe, chosen to improve your voice without hurting everyone else's:
- Only the top --train-layers of the 24 FastConformer encoder layers, plus the TDT decoder
  and joint network, are trained. The lower encoder (general acoustics) stays frozen, and so
  do all BatchNorm statistics, which small personal datasets would otherwise skew.
- General speech replay: LibriSpeech-PC utterances (punctuated) are mixed into every epoch,
  --replay x as much audio as your own clips, so the model keeps its general accuracy and
  its punctuation and capitalization.
- Low learning rate (default 2e-5, cosine decay, short warm-up), bf16 mixed precision,
  small batches plus gradient accumulation, and SpecAugment as configured in the base model.
- Your validation split picks the best checkpoint. The held-out test split is never
  loaded here.

Output: WORK/models/<name>/model.nemo plus train_info.json.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import BASE_MODEL, WORK, ensure_numba_cuda, read_manifest, write_manifest  # noqa: E402

OUT_OF_MEMORY = 75  # exit code: pipeline.sh retries with smaller batches


def build_train_manifest(user: list[dict], replay: list[dict], ratio: float, seed: int,
                         out: Path) -> dict:
    """Your clips plus a random slice of general speech, ``ratio`` x your audio duration."""
    rng = random.Random(seed)
    want = ratio * sum(r["duration"] for r in user)
    pool = replay[:]
    rng.shuffle(pool)
    picked, total = [], 0.0
    for r in pool:
        if total >= want:
            break
        picked.append(r)
        total += r["duration"]
    rows = user + picked
    rng.shuffle(rows)
    write_manifest(out, rows)
    return {"user_clips": len(user), "user_minutes": round(sum(r["duration"] for r in user) / 60, 2),
            "replay_clips": len(picked), "replay_minutes": round(total / 60, 2)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=WORK / "data")
    ap.add_argument("--name", default=time.strftime("tuned-%Y%m%d-%H%M"))
    ap.add_argument("--base", default=BASE_MODEL, help="HF name or a local .nemo file")
    ap.add_argument("--train-layers", type=int, default=6, help="top encoder layers to train (0-24)")
    ap.add_argument("--replay", type=float, default=1.0, help="general audio per unit of your audio")
    ap.add_argument("--epochs", type=float, default=8.0, help="passes over your clips")
    ap.add_argument("--steps", type=int, default=0, help="optimizer steps (overrides --epochs)")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--accumulate", type=int, default=4)
    ap.add_argument("--max-seconds", type=float, default=20.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--precision", default="bf16-mixed")
    ap.add_argument("--cpu", action="store_true", help="train on CPU (slow; pipeline checks only)")
    ap.add_argument("--tiny", action="store_true", help="small validation set, for quick checks")
    args = ap.parse_args()

    if not args.cpu:
        ensure_numba_cuda()  # before NeMo: its transducer loss is compiled by Numba

    import lightning.pytorch as pl
    import torch
    from lightning.pytorch.callbacks import Callback, ModelCheckpoint
    from omegaconf import open_dict

    import nemo.collections.asr as nemo_asr

    torch.manual_seed(args.seed)
    out = WORK / "models" / args.name
    out.mkdir(parents=True, exist_ok=True)
    user_train = read_manifest(args.data / "user" / "train.json")
    user_val = read_manifest(args.data / "user" / "val.json")
    replay = read_manifest(args.data / "general" / "replay.json") if args.replay > 0 else []
    if not user_train:
        raise SystemExit("No training clips: run prepare_data.py first.")
    mix = build_train_manifest(user_train, replay, args.replay, args.seed, out / "train_mix.json")
    val = user_val or user_train[: max(1, len(user_train) // 10)]
    if args.tiny:
        val = val[:16]
    write_manifest(out / "val.json", val)
    print(f"train mix: {mix}; val: {len(val)} clips")

    t0 = time.time()
    if args.base.endswith(".nemo"):
        model = nemo_asr.models.ASRModel.restore_from(args.base, map_location="cpu")
    else:
        model = nemo_asr.models.ASRModel.from_pretrained(args.base, map_location="cpu")
    print(f"loaded {args.base} in {time.time() - t0:.0f}s")

    common_ds = {
        "sample_rate": 16000, "max_duration": args.max_seconds, "min_duration": 0.3,
        "num_workers": args.workers, "pin_memory": True, "use_lhotse": False,
        "use_start_end_token": False, "trim_silence": False, "is_tarred": False,
    }
    with open_dict(model.cfg):
        model.cfg.train_ds = {**common_ds, "manifest_filepath": str(out / "train_mix.json"),
                              "batch_size": args.batch, "shuffle": True}
        model.cfg.validation_ds = {**common_ds, "manifest_filepath": str(out / "val.json"),
                                   "batch_size": args.batch, "shuffle": False}
    model.setup_training_data(model.cfg.train_ds)
    model.setup_multiple_validation_data(model.cfg.validation_ds)
    # CUDA-graph decoding keeps big extra buffers on the card; next to training's memory on an
    # 8 GB card that ran out at the first validation. Plain decoding is just as accurate.
    decoding = copy.deepcopy(model.cfg.decoding)
    with open_dict(decoding):
        decoding.greedy.use_cuda_graph_decoder = False
    model.change_decoding_strategy(decoding, verbose=False)

    n_rows = len(read_manifest(out / "train_mix.json"))
    steps_per_epoch = max(1, math.ceil(n_rows / (args.batch * args.accumulate)))
    user_share = mix["user_clips"] / max(1, n_rows)
    steps = args.steps or max(10, round(args.epochs * steps_per_epoch))  # epochs over the mix
    warmup = max(1, steps // 10)
    with open_dict(model.cfg):
        model.cfg.optim = {
            "name": "adamw", "lr": args.lr, "betas": [0.9, 0.98], "weight_decay": 1e-3,
            "sched": {"name": "CosineAnnealing", "warmup_steps": warmup, "min_lr": args.lr / 10,
                      "max_steps": steps},
        }

    # Freeze everything, then unfreeze the top encoder layers, decoder and joint.
    model.freeze()
    layers = model.encoder.layers
    k = max(0, min(args.train_layers, len(layers)))
    for layer in list(layers)[len(layers) - k:]:
        layer.requires_grad_(True)
    model.decoder.requires_grad_(True)
    model.joint.requires_grad_(True)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"training {trainable / 1e6:.1f}M of {total / 1e6:.1f}M parameters "
          f"(top {k}/{len(layers)} encoder layers + decoder + joint)")

    class KeepFrozenInEval(Callback):
        """Train mode for what's trained; frozen parts and all BatchNorm statistics stay fixed
        so a small dataset can't skew them.

        Lightning 2.2+ keeps each submodule's mode instead of calling train() on the whole
        model, so after freeze() the decoder's LSTM stayed in eval mode and the first
        backward pass failed ("cudnn RNN backward can only be called in training mode")."""

        def on_validation_start(self, trainer, pl_module):
            torch.cuda.empty_cache()  # hand training's spare memory back before decoding

        def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
            pl_module.train()
            for m in pl_module.modules():
                if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                    m.eval()
                elif any(True for _ in m.parameters(recurse=False)) and \
                        not any(p.requires_grad for p in m.parameters(recurse=False)):
                    m.eval()  # frozen leaf modules: no dropout either

    ckpt = ModelCheckpoint(dirpath=out / "ckpt", monitor="val_wer", mode="min", save_top_k=1,
                           save_weights_only=True, filename="best")
    val_every = max(1, min(steps, steps_per_epoch))
    trainer = pl.Trainer(
        accelerator="cpu" if args.cpu else "gpu", devices=1,
        precision="32-true" if args.cpu else args.precision,
        max_steps=steps, accumulate_grad_batches=args.accumulate, gradient_clip_val=1.0,
        val_check_interval=val_every * args.accumulate, check_val_every_n_epoch=None,
        logger=False, enable_checkpointing=True, callbacks=[KeepFrozenInEval(), ckpt],
        log_every_n_steps=5, num_sanity_val_steps=0, enable_progress_bar=True)
    model.set_trainer(trainer)
    if torch.cuda.is_available() and not args.cpu:
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    try:
        trainer.fit(model)
    except Exception as exc:  # noqa: BLE001
        if "out of memory" in str(exc).lower() and args.batch > 1 and not args.cpu:
            # pipeline.sh runs this again with half the batch and twice the accumulation:
            # the same effective batch in about half the graphics memory.
            print(f"Ran out of graphics memory with batches of {args.batch}; trying smaller "
                  "ones (other apps on this PC use some of the graphics card too)", flush=True)
            sys.exit(OUT_OF_MEMORY)
        raise
    train_s = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() and not args.cpu else 0
    if ckpt.best_model_path:
        state = torch.load(ckpt.best_model_path, map_location="cpu", weights_only=False)["state_dict"]
        model.load_state_dict(state)
        print(f"best checkpoint: val_wer {float(ckpt.best_model_score):.4f}")
    model.save_to(str(out / "model.nemo"))
    info = {"name": args.name, "base": args.base, "created": time.strftime("%Y-%m-%d %H:%M"),
            **mix, "steps": steps, "lr": args.lr, "train_layers": k, "batch": args.batch,
            "accumulate": args.accumulate, "replay_ratio": args.replay,
            "train_seconds": round(train_s), "peak_vram_gb": round(peak, 2),
            "best_val_wer": float(ckpt.best_model_score) if ckpt.best_model_score is not None else None,
            "user_share_of_mix": round(user_share, 3)}
    (out / "train_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info, indent=2))
    print(f"\nsaved {out / 'model.nemo'}\nnext: python export_onnx.py {out}")


if __name__ == "__main__":
    main()
