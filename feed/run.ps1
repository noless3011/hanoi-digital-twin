$ErrorActionPreference = 'Stop'
$workspaceRoot = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $workspaceRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $venvPython)) {
    throw 'Run setup.ps1 in the project root first.'
}
& $venvPython -u (Join-Path $PSScriptRoot 'video_adapter.py') @args
exit $LASTEXITCODE
