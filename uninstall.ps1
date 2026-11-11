# Removes Murmr from Windows: the app, its Python, shortcuts and launch at login.
# Your settings, dictionary, history and voice samples stay unless you pass -All.
#     powershell -ExecutionPolicy Bypass -File uninstall.ps1 [-All]
param([switch]$All)
$ErrorActionPreference = "SilentlyContinue"

# Stop Murmr (installed copies run <install folder>\boot.py).
Get-CimInstance Win32_Process |
    Where-Object { $_.CommandLine -match '\\(Murmr|Whispa)\\boot\.py' } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
Start-Sleep -Milliseconds 800

$run = "HKCU:\Software\Microsoft\Windows\CurrentVersion\Run"
$desktops = @([Environment]::GetFolderPath("Desktop"), (Join-Path $env:USERPROFILE "Desktop")) | Select-Object -Unique
foreach ($name in "Murmr", "Whispa") {  # Whispa: the earlier name
    Remove-ItemProperty -Path $run -Name $name
    Remove-Item (Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs\$name.lnk") -Force
    foreach ($d in $desktops) { Remove-Item (Join-Path $d "$name.lnk") -Force }
    $root = Join-Path $env:LOCALAPPDATA $name
    if (Test-Path (Join-Path $root "install.json")) { Remove-Item $root -Recurse -Force }
    if ($All) { Remove-Item (Join-Path $env:APPDATA $name) -Recurse -Force }
}
if ($All) { Write-Host "Murmr and all its data are removed." }
else { Write-Host "Murmr is removed. Your settings and history are kept (run with -All to remove them too)." }
