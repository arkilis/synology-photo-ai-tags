param(
    [string]$Root = "Z:\Photos",
    [string]$Progress = "",
    [string]$Model = "qwen2.5vl:7b",
    [string]$OllamaHost = "http://localhost:11434",
    [string]$ImageConverterBin = "magick",
    [int]$BatchSize = 4,
    [int]$RequestsPerMinute = 999,
    [int]$RequestTimeout = 600,
    [int]$MaxFilesPerRun = 0,
    [switch]$DryRun,
    [switch]$Force
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Write-Step {
    param([string]$Message)
    Write-Host "[run-windows] $Message"
}

function Refresh-ProcessPath {
    $machinePath = [Environment]::GetEnvironmentVariable("Path", "Machine")
    $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
    $parts = @($machinePath, $userPath) | Where-Object { $_ }
    $env:Path = ($parts -join ";")
}

function Find-CommandPath {
    param([string]$CommandName)
    $command = Get-Command $CommandName -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($null -ne $command) {
        return $command.Source
    }
    return $null
}

function Ensure-Winget {
    if (Find-CommandPath "winget") {
        return
    }
    throw "winget is required to auto-install dependencies. Install App Installer / winget first."
}

function Install-WithWinget {
    param(
        [string]$PackageId,
        [string]$Label
    )

    Write-Step "Installing $Label with winget ($PackageId)..."
    & winget install --exact --id $PackageId --accept-source-agreements --accept-package-agreements --disable-interactivity
}

function Ensure-Command {
    param(
        [string]$CommandName,
        [string]$PackageId,
        [string]$Label
    )

    $existing = Find-CommandPath $CommandName
    if ($existing) {
        Write-Step "$Label already installed: $existing"
        return $existing
    }

    Ensure-Winget
    Install-WithWinget -PackageId $PackageId -Label $Label
    Refresh-ProcessPath
    Start-Sleep -Seconds 2

    $installed = Find-CommandPath $CommandName
    if ($installed) {
        Write-Step "$Label installed: $installed"
        return $installed
    }

    throw "$Label installation completed but '$CommandName' is still not available in PATH. Open a new PowerShell window and try again."
}

function Get-PythonLauncher {
    if (Find-CommandPath "python") {
        return @("python")
    }
    if (Find-CommandPath "py") {
        return @("py", "-3")
    }
    throw "Python was not found. Install Python 3 and make sure 'python' or 'py' is available in PATH."
}

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $projectRoot

try {
    if ([string]::IsNullOrWhiteSpace($Progress)) {
        $Progress = Join-Path $Root ".ai-tags-progress.json"
    }

    Ensure-Command -CommandName "magick" -PackageId "ImageMagick.ImageMagick" -Label "ImageMagick"
    Ensure-Command -CommandName "exiftool" -PackageId "OliverBetz.ExifTool" -Label "ExifTool"

    $pythonLauncher = Get-PythonLauncher
    $pythonCommand = $pythonLauncher[0]
    $pythonPrefixArgs = @()
    if ($pythonLauncher.Count -gt 1) {
        $pythonPrefixArgs = $pythonLauncher[1..($pythonLauncher.Count - 1)]
    }

    $args = @(
        "-m", "src",
        "--backend", "ollama",
        "--model", $Model,
        "--ollama-host", $OllamaHost,
        "--image-converter-bin", $ImageConverterBin,
        "--batch-size", $BatchSize.ToString(),
        "--requests-per-minute", $RequestsPerMinute.ToString(),
        "--request-timeout", $RequestTimeout.ToString(),
        "--progress", $Progress,
        "--root", $Root
    )

    if ($MaxFilesPerRun -gt 0) {
        $args += @("--max-files-per-run", $MaxFilesPerRun.ToString())
    }
    if ($DryRun) {
        $args += "--dry-run"
    }
    if ($Force) {
        $args += "--force"
    }

    Write-Step "Starting photo tagging with Ollama model $Model"
    Write-Step "Root: $Root"
    Write-Step "Progress: $Progress"

    & $pythonCommand @pythonPrefixArgs @args
}
finally {
    Pop-Location
}
