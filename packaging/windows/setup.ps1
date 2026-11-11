# Run by Murmr-Setup.exe: the normal installer, keeping the window open if something fails.
$Host.UI.RawUI.WindowTitle = "Installing Murmr"
try {
    & (Join-Path $PSScriptRoot "install.ps1")
    Start-Sleep -Seconds 3
} catch {
    Write-Host "`n$_" -ForegroundColor Red
    Read-Host "`nMurmr couldn't be installed. Press Enter to close"
    exit 1
}
