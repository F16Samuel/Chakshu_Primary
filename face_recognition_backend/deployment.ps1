<#
.SYNOPSIS
    Launches the Campus Face Recognition System's Registration, Recognition, and Node.js Backend microservices.

.DESCRIPTION
    This script starts the 'registration' and 'recognition' FastAPI applications using Uvicorn,
    and the Node.js backend server. Each service is launched in its own new PowerShell window for easy monitoring.

.NOTES
    - Place this script in the root 'face/' directory of your project.
    - Ensure Python, Uvicorn, Node.js, and npm are installed globally or are accessible in your system's PATH.
    - Ensure all Python and Node.js dependencies are installed (e.g., via `pip install -r requirements.txt` for Python
      and `npm install` in the 'backend' directory for Node.js).
    - For production, remove the '--reload' flag from uvicorn commands and consider running Node.js with a process manager.
    - This script assumes PowerShell 5.1 or later.
#>

# --- Configuration ---
$ProjectRoot = Get-Item (Split-Path -Parent $MyInvocation.MyCommand.Definition)
$RegistrationServicePath = Join-Path $ProjectRoot "reg_server"
$RecognitionServicePath = Join-Path $ProjectRoot "check_server"
$NodeBackendPath = Join-Path $ProjectRoot "backend" # Path to your Node.js backend folder

Write-Host "Launching Campus Face Recognition System Microservices..." -ForegroundColor Green

# --- Launch Registration Service ---
$RegistrationCommand = @(
    "python -m uvicorn reg_server.registration:app",
    "--host 0.0.0.0",
    "--port 8000",
    "--reload", # Remove for production
    "--log-level info"
) -join " "

Write-Host "Starting Registration Service (Python FastAPI)..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit -Command `"Set-Location '$ProjectRoot'; $RegistrationCommand`""

# --- Launch Recognition Service ---
$RecognitionCommand = @(
    "python -m uvicorn check_server.recognition:app",
    "--host 0.0.0.0",
    "--port 8001",
    "--reload", # Remove for production
    "--log-level info"
) -join " "

Write-Host "Starting Recognition Service (Python FastAPI)..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit -Command `"Set-Location '$ProjectRoot'; $RecognitionCommand`""

# --- Launch Node.js Backend Server ---
# Assuming 'server.js' is the main entry point in the 'backend' directory
$NodeBackendCommand = "node server.js"

Write-Host "Starting Node.js Backend Server..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit -Command `"Set-Location '$NodeBackendPath'; $NodeBackendCommand`""

Write-Host "`nAll services launched. Check the new PowerShell windows for logs." -ForegroundColor Green
Write-Host "Press any key to close this launcher window (services will continue to run in their own windows)." -ForegroundColor Yellow
Pause | Out-Null