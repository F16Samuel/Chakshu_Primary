# --- Configuration ---
$ProjectRoot = Get-Item (Split-Path -Parent $MyInvocation.MyCommand.Definition)
$RegistrationServicePath = Join-Path $ProjectRoot "reg_server"
$RecognitionServicePath = Join-Path $ProjectRoot "check_server"
$NodeBackendPath = Join-Path $ProjectRoot "backend" # Path to your Node.js backend folder

Write-Host "Campus Face Recognition System Microservices Launcher" -ForegroundColor Green
Write-Host "=========================================================" -ForegroundColor Green

# Function to check and install Python dependencies
function Install-PythonDependencies {
    param(
        [string]$Directory,
        [string]$ServiceName
    )
    
    if (Test-Path "$Directory\requirements.txt") {
        Write-Host "Installing Python dependencies for $ServiceName..." -ForegroundColor Yellow
        Set-Location $Directory
        pip install -r requirements.txt
        Set-Location $ProjectRoot
        Write-Host "Python dependencies installed for $ServiceName" -ForegroundColor Green
    } else {
        Write-Host "No requirements.txt found for $ServiceName in $Directory" -ForegroundColor Gray
    }
}

# Function to check and install npm dependencies
function Install-NPMDependencies {
    param(
        [string]$Directory,
        [string]$ServiceName
    )
    
    if (Test-Path "$Directory\package.json") {
        if (-not (Test-Path "$Directory\node_modules")) {
            Write-Host "Installing npm dependencies for $ServiceName..." -ForegroundColor Yellow
            Set-Location $Directory
            npm install
            Set-Location $ProjectRoot
            Write-Host "NPM dependencies installed for $ServiceName" -ForegroundColor Green
        } else {
            Write-Host "NPM dependencies already exist for $ServiceName" -ForegroundColor Green
        }
    } else {
        Write-Host "No package.json found for $ServiceName in $Directory" -ForegroundColor Gray
    }
}

# Function to start a service in a new window
function Start-ServiceInNewWindow {
    param(
        [string]$Title,
        [string]$WorkingDirectory,
        [string]$Command
    )
    
    Write-Host "Starting $Title..." -ForegroundColor Cyan
    
    if (Test-Path $WorkingDirectory) {
        Start-Process powershell -ArgumentList "-NoExit -Command `"Set-Location '$WorkingDirectory'; $Command; Write-Host '$Title is running...' -ForegroundColor Green`""
        Start-Sleep -Seconds 2
    } else {
        Write-Host "Directory not found: $WorkingDirectory" -ForegroundColor Red
    }
}

# --- Install Dependencies ---
Write-Host "`n=== Installing Dependencies ===" -ForegroundColor Magenta

# Check for project-wide Python dependencies
Install-PythonDependencies -Directory $ProjectRoot -ServiceName "Project Root"

# Check for service-specific Python dependencies
Install-PythonDependencies -Directory $RegistrationServicePath -ServiceName "Registration Service"
Install-PythonDependencies -Directory $RecognitionServicePath -ServiceName "Recognition Service"

# Check for Node.js backend dependencies
Install-NPMDependencies -Directory $NodeBackendPath -ServiceName "Node.js Backend"

Write-Host "`n=== Starting Services ===" -ForegroundColor Magenta

# --- Launch Registration Service ---
$RegistrationCommand = @(
    "python -m uvicorn reg_server.registration:app",
    "--host 0.0.0.0",
    "--port 8000",
    "--reload", # Remove for production
    "--log-level info"
) -join " "

Start-ServiceInNewWindow -Title "Registration Service (FastAPI)" -WorkingDirectory $ProjectRoot -Command $RegistrationCommand

# --- Launch Recognition Service ---
$RecognitionCommand = @(
    "python -m uvicorn check_server.recognition:app",
    "--host 0.0.0.0",
    "--port 8001",
    "--reload", # Remove for production
    "--log-level info"
) -join " "

Start-ServiceInNewWindow -Title "Recognition Service (FastAPI)" -WorkingDirectory $ProjectRoot -Command $RecognitionCommand

# --- Launch Node.js Backend Server ---
# Assuming 'server.js' is the main entry point in the 'backend' directory
$NodeBackendCommand = "node server.js"

Start-ServiceInNewWindow -Title "Node.js Backend Server" -WorkingDirectory $NodeBackendPath -Command $NodeBackendCommand

# --- Service Information ---
Write-Host "`n=== All Services Launched ===" -ForegroundColor Green
Write-Host "Services are now running in separate PowerShell windows:" -ForegroundColor Cyan
Write-Host "  - Registration Service: http://localhost:8000" -ForegroundColor White
Write-Host "  - Recognition Service: http://localhost:8001" -ForegroundColor White
Write-Host "  - Node.js Backend: Check console output for port" -ForegroundColor White

Write-Host "`nAPI Documentation:" -ForegroundColor Cyan
Write-Host "  - Registration Service Docs: http://localhost:8000/docs" -ForegroundColor White
Write-Host "  - Recognition Service Docs: http://localhost:8001/docs" -ForegroundColor White

Write-Host "`nTo stop services, close their respective PowerShell windows." -ForegroundColor Yellow
Write-Host "Press any key to close this launcher window (services will continue to run)." -ForegroundColor Yellow

# Keep the main script running until user input
try {
    Read-Host "Press Enter to exit launcher"
} catch {
    Write-Host "`nLauncher closed. Services are still running in separate windows." -ForegroundColor Yellow
}
