param(
    [string]$ProjectRoot = "C:\Users\35135\Desktop\PHM"
)

$ErrorActionPreference = "Stop"

Write-Host "Checking Python executable..."
$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) {
    $python = Get-Command py -ErrorAction SilentlyContinue
}

if (-not $python) {
    Write-Host "Python was not found in PATH."
    Write-Host "Install Python 3.10+ and then run:"
    Write-Host "  cd $ProjectRoot"
    Write-Host "  python -m pip install -r requirements.txt"
    Write-Host "  python main.py"
    exit 1
}

Write-Host "Python found: $($python.Source)"
Set-Location $ProjectRoot
& $python.Source -m pip install -r requirements.txt
& $python.Source main.py
