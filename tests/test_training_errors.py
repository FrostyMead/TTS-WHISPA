"""When training fails, Murmr says why (not the progress bar redrawn after the crash), keeps
the whole output in a log, and the report's "Show the file" opens the right folder."""
import sys
import types
from pathlib import Path

import pytest

from whispa import tuning
from whispa.report import explorer_select

# The real end of the failed run on 2026-11-13 (shortened).
NVVM_FAILURE = [
    "== 3/6 Training on your voice",
    "Traceback (most recent call last):",
    'File "/home/me/whispa-finetune/.venv/lib/python3.12/site-packages/numba/cuda/cudadrv/nvvm.py", '
    "line 144, in __new__",
    "raise NvvmSupportError(errmsg % e)",
    "numba.cuda.cudadrv.error.NvvmSupportError: libNVVM cannot be found. Do `conda install "
    "cudatoolkit`:",
    "libnvvm.so: cannot open shared object file: No such file or directory",
    "Epoch 0:   0%|          | 0/36 [00:14<?, ?it/s]",
]


def test_the_error_is_shown_not_the_progress_bar():
    assert "NvvmSupportError: libNVVM cannot be found" in tuning.error_summary(NVVM_FAILURE, 1)


@pytest.mark.parametrize("lines, expected", [
    (["Epoch 0: 3%|▍ | 1/36", "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate"],
     "torch.OutOfMemoryError"),
    (["== 3/6 Training on your voice", "./pipeline.sh: line 17:  4242 Killed  python train.py"],
     "Killed"),
    (["== 1/6 Preparing your recordings", "Only 2 minutes of recordings; record more first."],
     "Only 2 minutes"),
    (["Epoch 0:   0%|          | 0/36"], "exit 1"),
    ([], "exit 1"),
])
def test_error_summary(lines, expected):
    assert expected in tuning.error_summary(lines, 1)


def _fake_job(monkeypatch, tmp_path, script: str) -> tuning.Job:
    monkeypatch.setattr("whispa.config.config_dir", lambda: tmp_path)
    monkeypatch.setattr(tuning, "CREATE_NO_WINDOW", 0)
    prog = tmp_path / "pipeline.py"
    prog.write_text(script, encoding="utf-8")
    job = tuning.Job("recognition")
    job.command = lambda: [sys.executable, str(prog)]
    return job


def test_a_failed_run_keeps_its_output_and_says_why(monkeypatch, tmp_path, caplog):
    job = _fake_job(monkeypatch, tmp_path, (
        "import sys\n"
        "print('== 3/6 Training on your voice', flush=True)\n"
        "sys.stdout.write('Epoch 0:   0%|          | 0/36\\r'); sys.stdout.flush()\n"
        "print('RuntimeError: libNVVM cannot be found', flush=True)\n"
        "sys.stdout.write('Epoch 0:   0%|          | 0/36 [00:14<?, ?it/s]')\n"
        "sys.exit(1)\n"))
    job._run()
    assert job.ok is False
    assert job.status == "Stopped with an error: RuntimeError: libNVVM cannot be found"
    log = (tmp_path / "tuning" / "recognition.log").read_text(encoding="utf-8")
    assert "== 3/6 Training on your voice" in log and "libNVVM" in log
    assert "0%|" not in log  # progress bars aren't kept
    assert "libNVVM" in caplog.text  # and the end goes into Murmr's own log, for reports


def test_a_successful_run(monkeypatch, tmp_path):
    job = _fake_job(monkeypatch, tmp_path, "print('== done tuned-20261113-1000')\n")
    job._run()
    assert job.ok and job.status == "Finished" and job.result == "tuned-20261113-1000"


def test_show_the_file_selects_a_name_with_spaces():
    path = Path(r"C:\Users\me\Desktop\Murmr report 2026-11-13 1005.zip")
    assert explorer_select(path) == \
        r'explorer.exe /select,"C:\Users\me\Desktop\Murmr report 2026-11-13 1005.zip"'


# --- the training scripts find NVIDIA's GPU compiler ---------------------------------------
def _common():
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts" / "finetune"))
    try:
        import common
    finally:
        sys.path.pop(0)
    return common


def test_numba_cuda_is_installed_with_numpy_and_cuda_held_back(monkeypatch, tmp_path):
    from importlib import metadata

    common = _common()
    monkeypatch.setitem(sys.modules, "numba_cuda", None)  # not installed
    monkeypatch.setattr(common, "WORK", tmp_path / "work")
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin" / "uv").write_text("")
    real = metadata.version
    monkeypatch.setattr(metadata, "version",
                        lambda name: "12.8.1" if name == "cuda-toolkit" else real(name))
    cmd = common.numba_cuda_install_command()
    assert cmd[:4] == [str(tmp_path / "bin" / "uv"), "pip", "install", "--python"]
    assert cmd[-3:] == ["numba-cuda[cu12]", "numpy<2.5", "cuda-toolkit==12.8.1"]


def test_nothing_to_install_when_numba_cuda_is_there(monkeypatch):
    common = _common()
    monkeypatch.setitem(sys.modules, "numba_cuda", types.ModuleType("numba_cuda"))
    assert common.numba_cuda_install_command() is None
