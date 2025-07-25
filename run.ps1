# PowerShell script to run all servers
# Author: Generated for multi-service deployment
# Description: Starts all frontend and backend services

Write-Host "Starting all servers..." -ForegroundColor Green

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
            Set-Location $currentDir
            Write-Host "NPM dependencies installed for $ServiceName" -ForegroundColor Green
        } else {
            Write-Host "NPM dependencies already exist for $ServiceName" -ForegroundColor Green
        }
    } else {
        Write-Host "No package.json found for $ServiceName in $Directory" -ForegroundColor Red
    }
}

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
        Set-Location $currentDir
        Write-Host "Python dependencies installed for $ServiceName" -ForegroundColor Green
    } else {
        Write-Host "No requirements.txt found for $ServiceName in $Directory" -ForegroundColor Gray
    }
}

# Function to start a process in a new window
function Start-ServiceInNewWindow {
    param(
        [string]$Title,
        [string]$WorkingDirectory,
        [string]$Command,
        [string]$Arguments = ""
    )
    
    Write-Host "Starting $Title..." -ForegroundColor Yellow
    
    if (Test-Path $WorkingDirectory) {
        $processArgs = @{
            FilePath = "powershell.exe"
            ArgumentList = @(
                "-NoExit",
                "-Command",
                "cd '$WorkingDirectory'; $Command $Arguments; Write-Host '$Title is running...' -ForegroundColor Green"
            )
            WindowStyle = "Normal"
        }
        Start-Process @processArgs
        Start-Sleep -Seconds 3
    } else {
        Write-Host "Directory not found: $WorkingDirectory" -ForegroundColor Red
    }
}

# Get current directory
$currentDir = Get-Location

Write-Host "Current directory: $currentDir" -ForegroundColor Cyan

# Frontend Services
Write-Host "`n=== Installing Frontend Dependencies ===" -ForegroundColor Magenta

# Install dependencies for all frontend services
Install-NPMDependencies -Directory "$currentDir\face_recognition_frontend" -ServiceName "Face Recognition Frontend"
Install-NPMDependencies -Directory "$currentDir\threat_detection_frontend" -ServiceName "Threat Detection Frontend"
Install-NPMDependencies -Directory "$currentDir\prim_frontend" -ServiceName "Prim Frontend"

# Backend Services Dependencies
Write-Host "`n=== Installing Backend Dependencies ===" -ForegroundColor Magenta

# Install Python dependencies for backend services
Install-PythonDependencies -Directory "$currentDir\face_recognition_backend" -ServiceName "Face Recognition Backend"
Install-PythonDependencies -Directory "$currentDir\threat_detection_backend" -ServiceName "Threat Detection Backend"

# Install Node.js dependencies for prim backend
Install-NPMDependencies -Directory "$currentDir\prim_backend" -ServiceName "Prim Backend"

Write-Host "`n=== Starting Frontend Services ===" -ForegroundColor Magenta

# Face Recognition Frontend (Port 8080)
Start-ServiceInNewWindow -Title "Face Recognition Frontend" -WorkingDirectory "$currentDir\face_recognition_frontend" -Command "npx" -Arguments "vite --port 8080"

# Threat Detection Frontend (Port 8081)
Start-ServiceInNewWindow -Title "Threat Detection Frontend" -WorkingDirectory "$currentDir\threat_detection_frontend" -Command "npx" -Arguments "vite --port 8081"

# Prim Frontend (Port 5173 - default Vite port)
Start-ServiceInNewWindow -Title "Prim Frontend" -WorkingDirectory "$currentDir\prim_frontend" -Command "npx" -Arguments "vite"

# Backend Services
Write-Host "`n=== Starting Backend Services ===" -ForegroundColor Magenta

# Face Recognition Backend (runs deployment.ps1)
Start-ServiceInNewWindow -Title "Face Recognition Backend" -WorkingDirectory "$currentDir\face_recognition_backend" -Command ".\deployment.ps1"

# Threat Detection Backend (FastAPI on port 8005)
Start-ServiceInNewWindow -Title "Threat Detection Backend" -WorkingDirectory "$currentDir\threat_detection_backend" -Command "python" -Arguments "app.py"

# Alternative for FastAPI with uvicorn (uncomment if needed)
# Start-ServiceInNewWindow -Title "Threat Detection Backend" -WorkingDirectory "$currentDir\threat_detection_backend" -Command "uvicorn" -Arguments "app:app --host 0.0.0.0 --port 8005"

# Prim Backend (Node.js)
Start-ServiceInNewWindow -Title "Prim Backend" -WorkingDirectory "$currentDir\prim_backend" -Command "node" -Arguments "index.js"

Write-Host "`n=== All services started! ===" -ForegroundColor Green
Write-Host "Frontend Services:" -ForegroundColor Cyan
Write-Host "  - Face Recognition: http://localhost:8080" -ForegroundColor White
Write-Host "  - Threat Detection: http://localhost:8081" -ForegroundColor White
Write-Host "  - Prim Frontend: http://localhost:5173" -ForegroundColor White

Write-Host "`nBackend Services:" -ForegroundColor Cyan
Write-Host "  - Face Recognition Backend: Check deployment.ps1 output for port" -ForegroundColor White
Write-Host "  - Threat Detection Backend: http://localhost:8005" -ForegroundColor White
Write-Host "  - Prim Backend: Check console output for port" -ForegroundColor White

Write-Host "`nPress Ctrl+C to stop this script. Individual services will continue running in their own windows." -ForegroundColor Yellow
Write-Host "To stop all services, close their respective PowerShell windows." -ForegroundColor Yellow

# Keep the main script running
try {
    while ($true) {
        Start-Sleep -Seconds 5
        # Optional: Add health checks here
    }
} catch {
    Write-Host "`nScript interrupted. Services are still running in separate windows." -ForegroundColor Yellow
}