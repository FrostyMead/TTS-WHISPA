"""Step 4: export a .nemo model to the ONNX layout onnx-asr (and so Whispa) loads.

    python export_onnx.py WORK/models/<name>            # uses <name>/model.nemo
    python export_onnx.py --base                        # the untouched base model (for A/B)

Produces <dir>/onnx/ with the same files as istupakov/parakeet-tdt-0.6b-v3-onnx:
  config.json                    {"model_type": "nemo-conformer-tdt", "features_size": 128, ...}
  encoder-model.onnx(+.data)     FastConformer encoder (weights > 2 GB live in the .data file)
  decoder_joint-model.onnx       TDT prediction network + joint
  vocab.txt                      "<token> <id>" per line, blank last
  *.int8.onnx                    dynamically quantized copies for CPU (unless --no-int8)
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import BASE_MODEL, WORK  # noqa: E402


def quantize(onnx_dir: Path) -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    for name in ("encoder-model", "decoder_joint-model"):
        src, dst = onnx_dir / f"{name}.onnx", onnx_dir / f"{name}.int8.onnx"
        quantize_dynamic(str(src), str(dst), weight_type=QuantType.QUInt8,
                         use_external_data_format=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model_dir", type=Path, nargs="?")
    ap.add_argument("--base", action="store_true", help="export the base model to WORK/models/base")
    ap.add_argument("--no-int8", action="store_true")
    args = ap.parse_args()

    import torch

    import nemo.collections.asr as nemo_asr

    t0 = time.time()
    if args.base:
        model_dir = WORK / "models" / "base"
        model_dir.mkdir(parents=True, exist_ok=True)
        model = nemo_asr.models.ASRModel.from_pretrained(BASE_MODEL, map_location="cpu")
    else:
        model_dir = args.model_dir
        model = nemo_asr.models.ASRModel.restore_from(str(model_dir / "model.nemo"), map_location="cpu")
    model.eval()
    onnx_dir = model_dir / "onnx"
    if onnx_dir.exists():
        shutil.rmtree(onnx_dir)
    onnx_dir.mkdir(parents=True)

    with torch.inference_mode():
        # RNNT/TDT models export as encoder-model.onnx + decoder_joint-model.onnx
        model.export(str(onnx_dir / "model.onnx"))
    with (onnx_dir / "vocab.txt").open("w", encoding="utf-8") as f:
        for i, token in enumerate([*model.tokenizer.vocab, "<blk>"]):
            f.write(f"{token} {i}\n")
    pre = model.cfg.preprocessor
    config = {"model_type": "nemo-conformer-tdt", "features_size": int(pre.features),
              "subsampling_factor": int(model.cfg.encoder.subsampling_factor)}
    (onnx_dir / "config.json").write_text(json.dumps(config, indent=4), encoding="utf-8")
    print(f"exported in {time.time() - t0:.0f}s: {sorted(p.name for p in onnx_dir.iterdir())}")
    if not args.no_int8:
        t1 = time.time()
        quantize(onnx_dir)
        print(f"int8 copies in {time.time() - t1:.0f}s")
    size = sum(p.stat().st_size for p in onnx_dir.iterdir()) / 1e9
    print(f"{onnx_dir}  ({size:.1f} GB)\nnext: python evaluate.py {model_dir}")


if __name__ == "__main__":
    main()
