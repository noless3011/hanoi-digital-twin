param([switch]$Cpu)

$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
Push-Location -LiteralPath $projectRoot
try {
    if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
        python -m venv .venv
        if ($LASTEXITCODE -ne 0) { throw 'Could not create .venv. Install Python 3.12 first.' }
    }
    $venvPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
    & $venvPython -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { throw 'pip upgrade failed.' }
    $torchIndex = 'https://download.pytorch.org/whl/cu126'
    $torchBuild = 'cu126'
    if ($Cpu) {
        $torchIndex = 'https://download.pytorch.org/whl/cpu'
        $torchBuild = 'cpu'
    }
    & $venvPython -m pip install "torch==2.7.1+$torchBuild" "torchvision==0.22.1+$torchBuild" --index-url $torchIndex
    if ($LASTEXITCODE -ne 0) { throw 'PyTorch installation failed.' }
    & $venvPython -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) { throw 'Tracking / feed dependency installation failed.' }

    $localNode = Join-Path $projectRoot '.venv\tools\node\node.exe'
    if (-not (Test-Path -LiteralPath $localNode)) {
        $nodeCommand = Get-Command node -ErrorAction SilentlyContinue
        if ($nodeCommand) {
            New-Item -ItemType Directory -Path (Split-Path $localNode) -Force | Out-Null
            Copy-Item -LiteralPath $nodeCommand.Source -Destination $localNode
        } else {
            Write-Warning 'YouTube requires Node.js >=22 or Deno >=2.3. Install one, then rerun setup.ps1.'
        }
    }
    Write-Host 'Setup complete. Simulation runs in Python (twin/); Unity only plays results back.'
    Write-Host 'Video adapter: .\feed\run.ps1 --source twin; tracking: .\perception\run_rfdetr.ps1'
} finally {
    Pop-Location
}
