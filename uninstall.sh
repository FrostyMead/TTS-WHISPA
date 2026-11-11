#!/usr/bin/env bash
# Removes Murmr from macOS / Linux: the app, its Python, launch at login.
# Your settings, dictionary, history and voice samples stay unless you pass --all.
#     curl -fsSL https://raw.githubusercontent.com/FrostyMead/murmr/main/uninstall.sh | bash
#     curl -fsSL https://raw.githubusercontent.com/FrostyMead/murmr/main/uninstall.sh | bash -s -- --all
set -u
ALL=""
[[ "${1:-}" == "--all" ]] && ALL=1

for name in Murmr Whispa; do  # Whispa: the earlier name
  lower=$(echo "$name" | tr '[:upper:]' '[:lower:]')
  if [[ "$(uname)" == "Darwin" ]]; then
    support="$HOME/Library/Application Support/$name"
    pkill -f "$support/runtime/boot.py" 2>/dev/null || true
    launchctl remove "com.$lower.app" 2>/dev/null || true
    rm -f "$HOME/Library/LaunchAgents/com.$lower.app.plist"
    rm -rf "$HOME/Applications/$name.app"
    [[ -f "$support/runtime/install.json" ]] && rm -rf "$support/runtime"
    [[ -n "$ALL" ]] && rm -rf "$support"
  else
    data="${XDG_DATA_HOME:-$HOME/.local/share}/$lower"
    pkill -f "$data/boot.py" 2>/dev/null || true
    rm -f "$HOME/.local/share/applications/$lower.desktop" "$HOME/.config/autostart/$lower.desktop"
    [[ -f "$data/install.json" ]] && rm -rf "$data"
    [[ -n "$ALL" ]] && rm -rf "${XDG_CONFIG_HOME:-$HOME/.config}/$lower"
  fi
done
if [[ -n "$ALL" ]]; then
  echo "Murmr and all its data are removed."
else
  echo "Murmr is removed. Your settings and history are kept (run with --all to remove them too)."
fi
