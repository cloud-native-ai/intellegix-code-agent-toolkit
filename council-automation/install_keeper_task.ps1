# install_keeper_task.ps1 — Register session_keeper.py as a Windows Task Scheduler task.
#
# Why: subprocess.Popen + DETACHED_PROCESS proved unreliable on Windows 11 — the
# pythonw.exe daemon exited within 30-60s of starting, even though Chrome (also
# detached) survived. Per 2026-05-22 Perplexity diagnosis, the likely cause is
# the Playwright Node.js subprocess seeing stdin EOF and exiting, which tears
# down Playwright IPC → asyncio task cancellation → asyncio.run unwind.
#
# Task Scheduler invokes pythonw.exe with its OWN stdio handling that doesn't
# suffer from the DETACHED_PROCESS stdio-EOF issue, AND can be configured to
# auto-restart on failure, run at boot, run as the logged-in user (so Chrome
# has access to the user profile), and survive logoff/login cycles.
#
# Run ONCE from an elevated PowerShell (Run as Administrator) the first time.
# Subsequent runs are idempotent — re-running updates the task in place.
#
# Usage:
#   PowerShell -ExecutionPolicy Bypass -File install_keeper_task.ps1
#   PowerShell -ExecutionPolicy Bypass -File install_keeper_task.ps1 -Uninstall

param(
    [switch]$Uninstall,
    [string]$TaskName = "PerplexitySessionKeeper",
    [string]$PythonW   = ""   # auto-detect if empty
)

$ErrorActionPreference = "Stop"

$keeperPath = "$HOME\.claude\council-automation\session_keeper.py"
$logDir     = "$HOME\.claude\logs"

if ($Uninstall) {
    if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
        Write-Host "Removing scheduled task '$TaskName'..."
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
        Write-Host "Done. Keeper will no longer auto-start at logon."
    } else {
        Write-Host "Task '$TaskName' not registered. Nothing to remove."
    }
    return
}

# Validate keeper script exists
if (-not (Test-Path $keeperPath)) {
    Write-Error "session_keeper.py not found at: $keeperPath"
    exit 1
}

# Find pythonw.exe (no-console Python interpreter)
if (-not $PythonW) {
    $pythonExe = (Get-Command python -ErrorAction SilentlyContinue).Source
    if (-not $pythonExe) {
        Write-Error "python not found on PATH. Pass -PythonW <path> explicitly."
        exit 1
    }
    $PythonW = Join-Path (Split-Path $pythonExe -Parent) "pythonw.exe"
    if (-not (Test-Path $PythonW)) {
        Write-Error "pythonw.exe not found next to python ($pythonExe). Pass -PythonW <path>."
        exit 1
    }
}
Write-Host "pythonw.exe: $PythonW"
Write-Host "keeper:      $keeperPath"

# Create log directory
New-Item -ItemType Directory -Path $logDir -Force | Out-Null

# Build the action: pythonw.exe -u session_keeper.py
$action = New-ScheduledTaskAction `
    -Execute $PythonW `
    -Argument "-u `"$keeperPath`"" `
    -WorkingDirectory (Split-Path $keeperPath -Parent)

# Triggers: at logon (so it starts when the user signs in) + every 8 minutes.
# Architecture pivot 2026-05-25: the keeper is now a ONE-SHOT script that exits
# after each refresh. Chrome stays alive across invocations via DETACHED_PROCESS.
# So TS repetition is what drives refresh, not asyncio.sleep inside long-lived
# pythonw. Interval picked at 8min because pplx.session-id has ~10-min effective
# TTL — verified empirically 2026-05-25 when 20-min interval caused mid-cycle
# session-validation hangs. Each reuse-path refresh is ~3s (no Chrome launch).
$logonTrigger = New-ScheduledTaskTrigger -AtLogOn -User "$env:USERNAME"
$periodicTrigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(2) `
    -RepetitionInterval (New-TimeSpan -Minutes 8) `
    -RepetitionDuration (New-TimeSpan -Days 365)

# Run as the current user (NOT SYSTEM — keeper needs user's Chrome profile + GUI session)
$principal = New-ScheduledTaskPrincipal `
    -UserId "$env:USERDOMAIN\$env:USERNAME" `
    -LogonType Interactive `
    -RunLevel Limited

# Settings: restart on failure, no time limit, allow on battery
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -RestartCount 5 `
    -RestartInterval (New-TimeSpan -Minutes 1) `
    -ExecutionTimeLimit (New-TimeSpan -Days 0)   # 0 = no time limit

# Remove existing task if present (idempotent install)
if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
    Write-Host "Removing existing task '$TaskName' before re-registering..."
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
}

Register-ScheduledTask `
    -TaskName $TaskName `
    -Description "Keeps a Perplexity browser session alive with fresh Cloudflare cookies. Exposes CDP on port 9222 so council_browser.py can attach in headless mode." `
    -Action $action `
    -Trigger @($logonTrigger, $periodicTrigger) `
    -Principal $principal `
    -Settings $settings | Out-Null

Write-Host ""
Write-Host "[OK] Task '$TaskName' registered."
Write-Host ""
Write-Host "Next steps:"
Write-Host "  1. Start the task now (one-time):"
Write-Host "       Start-ScheduledTask -TaskName $TaskName"
Write-Host "  2. Verify it's running:"
Write-Host "       Get-ScheduledTaskInfo -TaskName $TaskName"
Write-Host "       python `"$keeperPath`" --status"
Write-Host "  3. Watch the logs:"
Write-Host "       Get-Content `"$logDir\session_keeper.log`" -Wait -Tail 20"
Write-Host ""
Write-Host "Uninstall later with:"
Write-Host "  PowerShell -ExecutionPolicy Bypass -File `"$PSCommandPath`" -Uninstall"
