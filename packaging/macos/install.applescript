-- "Install Murmr" in Murmr-macOS.dmg: runs the installer in Terminal so you can watch it.
on run
	set msg to "Murmr is private voice dictation that runs on this Mac." & return & return & "The installer downloads Murmr, its own Python and the speech model (about 1 GB), so it takes a few minutes. A Terminal window shows the progress. No admin password needed." & return & return & "Afterwards, allow Murmr when macOS asks (Microphone, Input Monitoring, Accessibility)."
	display dialog msg buttons {"Cancel", "Install"} default button "Install" with title "Install Murmr" with icon note
	tell application "Terminal"
		activate
		do script "curl -fsSL https://raw.githubusercontent.com/FrostyMead/murmr/main/install.sh | bash"
	end tell
end run
