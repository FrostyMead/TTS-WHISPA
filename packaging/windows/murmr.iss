; Murmr-Setup.exe: a small per-user installer (no admin) built by .github/workflows/release.yml.
; It runs install.ps1, which fetches the newest tested version of Murmr, its own Python and
; the speech model; Murmr keeps itself up to date after that. Uninstall from Settings > Apps.

#define AppVersion GetEnv("MURMR_VERSION")
#if AppVersion == ""
  #define AppVersion "0.0.0"
#endif

[Setup]
AppId={{6E4B1F0A-3C2D-4B8E-9A71-5D0C2F8E4A13}
AppName=Murmr
AppVersion={#AppVersion}
AppVerName=Murmr
AppPublisher=FrostyMead
AppPublisherURL=https://github.com/FrostyMead/murmr
AppSupportURL=https://github.com/FrostyMead/murmr/issues
DefaultDirName={localappdata}\Murmr Setup
PrivilegesRequired=lowest
DisableDirPage=yes
DisableProgramGroupPage=yes
DisableReadyPage=yes
OutputDir=..\..\dist
OutputBaseFilename=Murmr-Setup
SetupIconFile=murmr.ico
UninstallDisplayIcon={app}\murmr.ico
UninstallDisplayName=Murmr
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
CloseApplications=no

[Messages]
WelcomeLabel2=This installs Murmr, private voice dictation that runs on your computer.%n%nIt downloads Murmr, its own copy of Python and the speech model (about 1 GB), so it takes a few minutes. A window shows the progress.%n%nNo admin rights needed.

[Files]
Source: "..\..\install.ps1"; DestDir: "{app}"; Flags: ignoreversion
Source: "setup.ps1"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\..\uninstall.ps1"; DestDir: "{app}"; Flags: ignoreversion
Source: "murmr.ico"; DestDir: "{app}"; Flags: ignoreversion

[Run]
Filename: "powershell.exe"; Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\setup.ps1"""; StatusMsg: "Installing Murmr (a few minutes)..."; Flags: waituntilterminated

[UninstallRun]
Filename: "powershell.exe"; Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\uninstall.ps1"""; Flags: waituntilterminated runhidden; RunOnceId: "RemoveMurmr"
