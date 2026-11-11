-- "Uninstall Murmr" in Murmr-macOS.dmg.
on run
	set msg to "Remove Murmr from this Mac?" & return & return & "Keep my data: your settings, dictionary and history stay, in case you reinstall."
	set choice to button returned of (display dialog msg buttons {"Cancel", "Remove everything", "Keep my data"} default button "Keep my data" with title "Uninstall Murmr" with icon caution)
	set extra to ""
	if choice is "Remove everything" then set extra to " -s -- --all"
	do shell script "curl -fsSL https://raw.githubusercontent.com/FrostyMead/murmr/main/uninstall.sh | bash" & extra
	display dialog "Murmr is removed." & return & return & "You can also remove Murmr from System Settings > Privacy & Security (Microphone, Input Monitoring, Accessibility)." buttons {"OK"} default button "OK" with title "Uninstall Murmr"
end run
