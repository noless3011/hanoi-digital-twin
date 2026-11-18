$ErrorActionPreference = 'Stop'
$inferenceWorkspace = Split-Path $PSScriptRoot -Parent
$inferencePython = Join-Path $inferenceWorkspace '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $inferencePython)) {
    throw 'Run .\setup.ps1 first.'
}
& $inferencePython -u (Join-Path $PSScriptRoot 'run_rfdetr.py') @args
exit $LASTEXITCODE
