# Murmr installer for Windows. No admin rights needed.
#
# Install (or re-install) from GitHub, with in-app updates:
#     irm https://raw.githubusercontent.com/FrostyMead/murmr/main/install.ps1 | iex
#
# Developer setup inside a git checkout (editable install, no updater):
#     powershell -ExecutionPolicy Bypass -File install.ps1 [-Runtime cuda|directml|cpu]
#
# Options: -Runtime auto|cuda|directml|cpu   -User (force a normal install from a checkout)
param(
    [ValidateSet("auto", "cuda", "directml", "cpu")][string]$Runtime = "auto",
    [switch]$User,
    [string]$Repo = "FrostyMead/murmr",
    [string]$Branch = "main"
)
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"  # makes Invoke-WebRequest much faster

function Step($msg) { Write-Host "==> $msg" -ForegroundColor Cyan }

# Run a program; fail only on a non-zero exit code. (Windows PowerShell can turn a tool's
# ordinary progress messages on stderr into errors, which would stop the install.)
function Run {
    $exe, $rest = $args
    $old = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try { & $exe @rest 2>&1 | ForEach-Object { Write-Host "    $_" } } finally { $ErrorActionPreference = $old }
    if ($LASTEXITCODE) { throw "$(Split-Path $exe -Leaf) failed (exit code $LASTEXITCODE)" }
}

if ($Runtime -eq "auto") {
    $Runtime = "cpu"
    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) { $Runtime = "cuda" }
}

$checkout = $PSScriptRoot -and (Test-Path (Join-Path $PSScriptRoot "pyproject.toml"))

# ---------------------------------------------------------------------------- uv
$uv = (Get-Command uv -ErrorAction SilentlyContinue).Source
if (-not $uv) {
    Step "Installing uv (Python manager, into your user folder)"
    Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
    $uv = Join-Path $env:USERPROFILE ".local\bin\uv.exe"
    if (-not (Test-Path $uv)) { throw "uv install failed" }
}

# ---------------------------------------------------------------------------- developer setup
if ($checkout -and -not $User) {
    Set-Location $PSScriptRoot
    Step "Developer setup in $PSScriptRoot ($Runtime)"
    if (-not (Test-Path .venv)) { Run $uv venv .venv --python 3.13 --python-preference only-managed }
    Run $uv pip install --python .venv -e ".[$Runtime,dev]"
    Run .venv\Scripts\python.exe -m whispa --download-model
    Run .venv\Scripts\python.exe -m whispa --install-shortcuts
    Write-Host "`nDone. Start Murmr from the Start Menu. (Developer copy: update it with git.)"
    return
}

# ---------------------------------------------------------------------------- normal install
$Root = Join-Path $env:LOCALAPPDATA "Murmr"
$OldRoot = Join-Path $env:LOCALAPPDATA "Whispa"
if (-not (Test-Path $Root) -and (Test-Path (Join-Path $OldRoot "install.json"))) {
    $Root = $OldRoot  # installed before the rename: update that copy in place
}
$App = Join-Path $Root "app"
$Venv = Join-Path $Root "venv"
Step "Installing Murmr to $Root ($Runtime)"
New-Item -ItemType Directory -Force $App | Out-Null

# Stop a running Murmr so its files can be replaced.
Get-CimInstance Win32_Process |
    Where-Object { $_.CommandLine -and ($_.CommandLine -match "boot\.py|launcher\.pyw|-m whispa") -and $_.Name -match "python" } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }

# Newest commit whose tests passed on GitHub (falls back to the branch head).
$headers = @{ "User-Agent" = "Murmr-Installer" }
$sha = $null
try {
    $runs = Invoke-RestMethod -Headers $headers "https://api.github.com/repos/$Repo/actions/workflows/ci.yml/runs?branch=$Branch&status=success&event=push&per_page=1"
    if ($runs.workflow_runs.Count -gt 0) { $sha = $runs.workflow_runs[0].head_sha; $date = $runs.workflow_runs[0].head_commit.timestamp }
} catch { }
if (-not $sha) {
    $head = Invoke-RestMethod -Headers $headers "https://api.github.com/repos/$Repo/commits/$Branch"
    $sha = $head.sha; $date = $head.commit.committer.date
}
Step "Downloading version $($sha.Substring(0, 7))"
$tmp = Join-Path $env:TEMP "whispa-install-$([guid]::NewGuid().ToString('N'))"
New-Item -ItemType Directory -Force $tmp | Out-Null
$zip = Join-Path $tmp "whispa.zip"
Invoke-WebRequest -Headers $headers "https://codeload.github.com/$Repo/zip/$sha" -OutFile $zip
Expand-Archive $zip -DestinationPath $tmp
$src = Get-ChildItem $tmp -Directory | Select-Object -First 1
# JSON without a byte-order mark (Windows PowerShell's utf8 adds one, which Python rejects).
function Write-Json($obj, $path) { [IO.File]::WriteAllText($path, ($obj | ConvertTo-Json), [Text.UTF8Encoding]::new($false)) }
Write-Json @{ sha = $sha; date = $date; repo = $Repo; branch = $Branch; changes = @() } (Join-Path $src.FullName "version.json")

$current = Join-Path $App "current"
$previous = Join-Path $App "previous"
if (Test-Path $current) {
    if (Test-Path $previous) { Remove-Item -Recurse -Force $previous }
    Move-Item $current $previous
}
Move-Item $src.FullName $current
Remove-Item -Recurse -Force $tmp
Remove-Item -Recurse -Force (Join-Path $App "next") -ErrorAction SilentlyContinue
Remove-Item -Force (Join-Path $Root "trial") -ErrorAction SilentlyContinue

Step "Setting up Python and packages (first time: a few minutes)"
if (-not (Test-Path (Join-Path $Venv "Scripts\python.exe"))) {
    # uv's own Python, so removing or upgrading a system Python can't break Murmr.
    Run $uv venv $Venv --python 3.13 --python-preference only-managed
}
Run $uv pip install --python (Join-Path $Venv "Scripts\python.exe") -r (Join-Path $current "pyproject.toml") --extra $Runtime

Copy-Item (Join-Path $current "whispa\_boot.py") (Join-Path $Root "boot.py") -Force
Write-Json @{ repo = $Repo; branch = $Branch; runtime = $Runtime; uv = $uv; installed = (Get-Date -Format o) } (Join-Path $Root "install.json")

# The base interpreter's pythonw.exe: virtual-env launcher stubs can open a console window.
$base = (& (Join-Path $Venv "Scripts\python.exe") -c "import sys; print(sys.base_prefix)").Trim()
$python = Join-Path $base "python.exe"
$pythonw = Join-Path $base "pythonw.exe"
$boot = Join-Path $Root "boot.py"

Step "Downloading the speech model (first time: ~650 MB)"
Run $python -I -S $boot --download-model
Step "Adding Murmr to the Start Menu and Desktop"
Run $python -I -S $boot --install-shortcuts
Run $python -I -S $boot --autostart on

Start-Process $pythonw -ArgumentList "-I", "-S", "`"$boot`"" -WorkingDirectory $Root
Write-Host "`nMurmr is installed and running (tray icon). It starts at login; Settings > Updates keeps it current." -ForegroundColor Green
