# Run this script to start the Project Brain backend server
# Make sure you've set up the environment first (see SETUP.md)

Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "  PROJECT BRAIN - Starting Backend Server" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host ""

# Check if venv exists
if (-Not (Test-Path "venv")) {
    Write-Host "ERROR: Virtual environment not found!" -ForegroundColor Red
    Write-Host "Please run setup first:" -ForegroundColor Yellow
    Write-Host "  python -m venv venv" -ForegroundColor Yellow
    Write-Host "  venv\Scripts\activate" -ForegroundColor Yellow
    Write-Host "  pip install -r requirements.txt" -ForegroundColor Yellow
    exit 1
}

# Check if .env exists
if (-Not (Test-Path ".env")) {
    Write-Host "WARNING: .env file not found!" -ForegroundColor Yellow
    Write-Host "Copying .env.example to .env..." -ForegroundColor Yellow
    Copy-Item ".env.example" ".env"
    Write-Host "Please edit .env and add your GOOGLE_API_KEY" -ForegroundColor Yellow
    Write-Host ""
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Green
& "venv\Scripts\Activate.ps1"

# Check if GOOGLE_API_KEY is set
$envContent = Get-Content ".env" -Raw
if ($envContent -match "GOOGLE_API_KEY=your_gemini_api_key_here" -or $envContent -notmatch "GOOGLE_API_KEY=") {
    Write-Host ""
    Write-Host "ERROR: GOOGLE_API_KEY not configured!" -ForegroundColor Red
    Write-Host "Please edit .env file and add your Gemini API key" -ForegroundColor Yellow
    Write-Host "Get your key from: https://makersuite.google.com/app/apikey" -ForegroundColor Yellow
    Write-Host ""
    exit 1
}

Write-Host ""
Write-Host "Starting FastAPI server..." -ForegroundColor Green
Write-Host "API will be available at:" -ForegroundColor Cyan
Write-Host "  - Main API: http://localhost:8000" -ForegroundColor White
Write-Host "  - API Docs: http://localhost:8000/docs" -ForegroundColor White
Write-Host "  - Health Check: http://localhost:8000/health" -ForegroundColor White
Write-Host ""
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host ""

# Run the server
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
