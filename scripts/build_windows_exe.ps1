Param(
    [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path "$PSScriptRoot\..").Path
Set-Location $ProjectRoot

Write-Host "==> Building VideoChangeDetector.exe" -ForegroundColor Cyan

if (-not (Get-Command $PythonExe -ErrorAction SilentlyContinue)) {
    throw "Python executable '$PythonExe' not found."
}

& $PythonExe -m venv venv
& .\venv\Scripts\python.exe -m pip install --upgrade pip
& .\venv\Scripts\pip.exe install -r requirements-desktop.txt
& .\venv\Scripts\pip.exe install pyinstaller

& .\venv\Scripts\pyinstaller.exe --noconfirm --clean --windowed --onefile `
    --name VideoChangeDetector `
    --collect-all PySide6 `
    --collect-all cv2 `
    --add-data "config.yaml;." `
    video_compare_desktop.py

Write-Host ""
Write-Host "Done. EXE location:" -ForegroundColor Green
Write-Host "$ProjectRoot\dist\VideoChangeDetector.exe"
