# Schedule the desk's daily cycle with Windows Task Scheduler.
# Runs every day at 00:10, after the last hour of yesterday has settled.
#
# Run once in PowerShell from the project folder:
#     powershell -ExecutionPolicy Bypass -File scripts\schedule_windows.ps1
#
# Remove it later with:
#     Unregister-ScheduledTask -TaskName "BatteryDesk" -Confirm:$false
#
# Note: your laptop must be on (or asleep with wake timers allowed) at 00:10.

$Project = Split-Path -Parent $PSScriptRoot
$Python  = Join-Path $Project ".venv\Scripts\python.exe"
$Script  = Join-Path $Project "scripts\desk.py"
$Log     = Join-Path $Project "logs\scheduler.log"

New-Item -ItemType Directory -Force -Path (Join-Path $Project "logs") | Out-Null

$Action  = New-ScheduledTaskAction -Execute "cmd.exe" `
           -Argument "/c `"`"$Python`" `"$Script`" run >> `"$Log`" 2>&1`"" `
           -WorkingDirectory $Project
$Trigger = New-ScheduledTaskTrigger -Daily -At 00:10
$Settings = New-ScheduledTaskSettingsSet -WakeToRun -StartWhenAvailable `
            -ExecutionTimeLimit (New-TimeSpan -Hours 1)

Register-ScheduledTask -TaskName "BatteryDesk" -Action $Action -Trigger $Trigger `
    -Settings $Settings -Description "Alberta battery trading desk: daily cycle" -Force

Write-Host "Scheduled 'BatteryDesk' daily at 00:10. Log: $Log"
Write-Host "AESO_API_KEY must be set as a USER environment variable (setx), not just in this window."
