<#
.SYNOPSIS
Builds Local AI Studio for the fastest hardware this computer has.

.DESCRIPTION
With an NVIDIA graphics card and the CUDA toolkit installed, the built-in engine is built
with CUDA and runs on the card. Otherwise it runs on the processor (CPU).

Run it from the repository folder:

    powershell -ExecutionPolicy Bypass -File scripts\build.ps1

Other arguments are passed on to "cargo build".

.PARAMETER Cpu
Leave CUDA out even when an NVIDIA card is found.
#>
param(
    [switch]$Cpu,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$CargoArgs = @()
)

$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)

function Test-Command([string]$Name) {
    return [bool](Get-Command $Name -ErrorAction SilentlyContinue)
}

# nvcc needs the Visual C++ compiler (cl.exe), which is only on PATH in a "Developer
# PowerShell" window. Elsewhere, load the environment that vcvars64.bat sets up.
function Import-VisualStudioEnvironment {
    if (Test-Command 'cl.exe') { return $true }
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
    if (-not (Test-Path $vswhere)) { return $false }
    $vs = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if (-not $vs) { return $false }
    $vcvars = Join-Path $vs 'VC\Auxiliary\Build\vcvars64.bat'
    if (-not (Test-Path $vcvars)) { return $false }
    cmd /c "`"$vcvars`" >nul 2>&1 && set" | ForEach-Object {
        if ($_ -match '^([^=]+)=(.*)$') {
            Set-Item -LiteralPath "env:$($Matches[1])" -Value $Matches[2]
        }
    }
    return (Test-Command 'cl.exe')
}

if (-not (Test-Command 'cargo')) {
    Write-Host "Rust isn't installed. Install it from https://rustup.rs, open a new window and run this again." -ForegroundColor Red
    exit 1
}

$features = @()
$target = 'the processor (CPU)'
$advice = ''

$hasNvidia = $false
if (-not $Cpu -and (Test-Command 'nvidia-smi')) {
    & nvidia-smi -L *> $null
    $hasNvidia = ($LASTEXITCODE -eq 0)
}

if ($hasNvidia) {
    # The CUDA installer sets CUDA_PATH; its bin folder isn't always on PATH yet.
    if (-not (Test-Command 'nvcc') -and $env:CUDA_PATH -and (Test-Path (Join-Path $env:CUDA_PATH 'bin\nvcc.exe'))) {
        $env:PATH = (Join-Path $env:CUDA_PATH 'bin') + ';' + $env:PATH
    }
    if (-not (Test-Command 'nvcc')) {
        $advice = @"
This computer has an NVIDIA graphics card, but the CUDA toolkit isn't installed, so the
built-in engine will use the processor. To use the card, install the toolkit from
https://developer.nvidia.com/cuda-downloads and run this script again.
(Ollama, if you use it, runs on the graphics card either way.)
"@
    }
    elseif (-not (Import-VisualStudioEnvironment)) {
        $advice = @"
The CUDA toolkit needs the Visual Studio C++ compiler, which wasn't found, so the built-in
engine will use the processor. Install the Visual Studio Build Tools with "Desktop
development with C++" (or run this from "x64 Native Tools PowerShell for VS"), then run
this script again.
"@
    }
    else {
        $features = @('--features', 'cuda')
        $target = 'the NVIDIA GPU (CUDA)'
    }
}

Write-Host "Building Local AI Studio. The built-in engine will run on $target."
if ($features.Count -gt 0) {
    Write-Host "CUDA kernels are compiled for this computer's graphics card; the first build takes a while."
}
Write-Host ''

& cargo build --release @features @CargoArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ''
Write-Host 'Done. Start it with:'
Write-Host ''
Write-Host '  .\target\release\local-ai-studio.exe'
if ($advice) {
    Write-Host ''
    Write-Host $advice
}
