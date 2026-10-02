$ws = New-Object -ComObject WScript.Shell
$shortcut = $ws.CreateShortcut("$env:APPDATA\Microsoft\Windows\Start Menu\Programs\Startup\SoruBot.lnk")
$shortcut.TargetPath = "C:\Users\etemk\soru-bot\start_bot.bat"
$shortcut.WorkingDirectory = "C:\Users\etemk\soru-bot"
$shortcut.WindowStyle = 7
$shortcut.Save()
Write-Host "Startup kisayolu olusturuldu."
