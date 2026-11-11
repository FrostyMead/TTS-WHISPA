#!/usr/bin/env bash
# Whispa installer for macOS and Linux. No admin rights needed.
#
# Install (or re-install) from GitHub, with in-app updates:
#     curl -fsSL https://raw.githubusercontent.com/FrostyMead/TTS-WHISPA/main/install.sh | bash
#
# Developer setup inside a git checkout (editable install, no updater):
#     ./install.sh [cuda|cpu]
#
# Environment overrides: WHISPA_RUNTIME=cpu|cuda  WHISPA_REPO=owner/name  WHISPA_BRANCH=main
#                        WHISPA_USER_INSTALL=1 (normal install even from a checkout)
set -euo pipefail

REPO="${WHISPA_REPO:-FrostyMead/TTS-WHISPA}"
BRANCH="${WHISPA_BRANCH:-main}"
RUNTIME="${WHISPA_RUNTIME:-${1:-auto}}"
if [[ "$RUNTIME" == "auto" ]]; then
  RUNTIME="cpu"   # macOS: CPU (fast on Apple Silicon); Linux: CUDA when an NVIDIA GPU is present
  if [[ "$(uname)" == "Linux" ]] && command -v nvidia-smi >/dev/null 2>&1; then RUNTIME="cuda"; fi
fi
step() { printf '\033[36m==> %s\033[0m\n' "$*"; }

# ---------------------------------------------------------------------------- uv
UV="$(command -v uv || true)"
if [[ -z "$UV" ]]; then
  step "Installing uv (Python manager, into your user folder)"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  UV="$HOME/.local/bin/uv"
  [[ -x "$UV" ]] || { echo "uv install failed"; exit 1; }
fi

# ---------------------------------------------------------------------------- developer setup
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" 2>/dev/null && pwd || true)"
if [[ -n "$SCRIPT_DIR" && -f "$SCRIPT_DIR/pyproject.toml" && -z "${WHISPA_USER_INSTALL:-}" ]]; then
  cd "$SCRIPT_DIR"
  step "Developer setup in $SCRIPT_DIR ($RUNTIME)"
  [[ -d .venv ]] || "$UV" venv .venv --python 3.13 --python-preference only-managed
  "$UV" pip install --python .venv -e ".[$RUNTIME,dev]"
  .venv/bin/python -m whispa --download-model
  .venv/bin/python -m whispa --install-shortcuts || true
  echo
  echo "Done. Start with: .venv/bin/whispa   (developer copy: update it with git)"
  exit 0
fi

# ---------------------------------------------------------------------------- normal install
if [[ "$(uname)" == "Darwin" ]]; then
  ROOT="$HOME/Library/Application Support/Whispa/runtime"
else
  ROOT="${XDG_DATA_HOME:-$HOME/.local/share}/whispa"
fi
APP="$ROOT/app"
VENV="$ROOT/venv"
step "Installing Whispa to $ROOT ($RUNTIME)"
mkdir -p "$APP"

# Stop a running Whispa so its files can be replaced.
pkill -f "$ROOT/boot.py" 2>/dev/null || true

step "Setting up Python (first time: a minute)"
[[ -x "$VENV/bin/python" ]] || "$UV" venv "$VENV" --python 3.13 --python-preference only-managed
PY="$VENV/bin/python"

# Newest commit whose tests passed on GitHub (falls back to the branch head); download it.
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
"$PY" - "$REPO" "$BRANCH" "$TMP" <<'PYCODE'
import io, json, sys, urllib.request, zipfile
repo, branch, tmp = sys.argv[1:4]
def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Whispa-Installer"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()
sha = date = None
try:
    runs = json.loads(get(f"https://api.github.com/repos/{repo}/actions/workflows/ci.yml/runs"
                          f"?branch={branch}&status=success&event=push&per_page=1"))
    if runs.get("workflow_runs"):
        sha = runs["workflow_runs"][0]["head_sha"]
        date = runs["workflow_runs"][0].get("head_commit", {}).get("timestamp", "")
except Exception:
    pass
if not sha:
    head = json.loads(get(f"https://api.github.com/repos/{repo}/commits/{branch}"))
    sha, date = head["sha"], head["commit"]["committer"]["date"]
print(f"==> Downloading version {sha[:7]}", flush=True)
zipfile.ZipFile(io.BytesIO(get(f"https://codeload.github.com/{repo}/zip/{sha}"))).extractall(tmp + "/src")
import os
top = os.path.join(tmp, "src", os.listdir(os.path.join(tmp, "src"))[0])
with open(os.path.join(top, "version.json"), "w") as f:
    json.dump({"sha": sha, "date": date, "repo": repo, "branch": branch, "changes": []}, f)
PYCODE
SRC="$(ls -d "$TMP"/src/*/ | head -1)"
SRC="${SRC%/}"

CURRENT="$APP/current"
if [[ -d "$CURRENT" ]]; then
  rm -rf "$APP/previous"
  mv "$CURRENT" "$APP/previous"
fi
mv "$SRC" "$CURRENT"
rm -rf "$APP/next" "$ROOT/trial"

step "Installing packages (first time: a few minutes)"
"$UV" pip install --python "$PY" -r "$CURRENT/pyproject.toml" --extra "$RUNTIME"

cp "$CURRENT/whispa/_boot.py" "$ROOT/boot.py"
"$PY" - "$ROOT/install.json" "$REPO" "$BRANCH" "$RUNTIME" "$UV" <<'PYCODE'
import json, sys, time
path, repo, branch, runtime, uv = sys.argv[1:6]
json.dump({"repo": repo, "branch": branch, "runtime": runtime, "uv": uv,
           "installed": time.strftime("%Y-%m-%dT%H:%M:%S")}, open(path, "w"), indent=2)
PYCODE

BASE="$("$PY" -c 'import sys; print(sys.base_prefix)')"
BOOTPY="$BASE/bin/python3"
BOOT="$ROOT/boot.py"

step "Downloading the speech model (first time: ~650 MB)"
"$BOOTPY" -I -S "$BOOT" --download-model
step "Adding Whispa to your applications"
"$BOOTPY" -I -S "$BOOT" --install-shortcuts
"$BOOTPY" -I -S "$BOOT" --autostart on >/dev/null

if [[ "$(uname)" == "Darwin" ]]; then
  open "$HOME/Applications/Whispa.app"
  cat <<'EOF'

Whispa is installed: ~/Applications/Whispa.app (menu-bar icon, starts at login).
Whispa's icon is in the menu bar (top right). Before the first dictation:
  • Allow Whispa when macOS asks (Microphone, Input Monitoring, Accessibility),
    or in Whispa's Settings > Permissions.
  • System Settings > Keyboard > "Press 🌐 key to" > Do Nothing
Settings > Updates keeps it current.
EOF
else
  nohup "$BOOTPY" -I -S "$BOOT" >/dev/null 2>&1 &
  echo
  echo "Whispa is installed and running. It's in your app launcher and starts at login."
fi
