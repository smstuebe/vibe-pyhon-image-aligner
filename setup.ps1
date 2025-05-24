# setup.ps1
# Script to initialize the tomato plant time-lapse project after checkout

# Display welcome message
Write-Host "Setting up tomato plant time-lapse project..." -ForegroundColor Green

# Create necessary folders if they don't exist
Write-Host "Creating folders..." -ForegroundColor Cyan
if (-Not (Test-Path -Path "img")) {
    New-Item -ItemType Directory -Path "img" | Out-Null
    Write-Host "  Created img folder."
}
else {
    Write-Host "  img folder already exists."
}

if (-Not (Test-Path -Path "processed")) {
    New-Item -ItemType Directory -Path "processed" | Out-Null
    Write-Host "  Created processed folder."
}
else {
    Write-Host "  processed folder already exists."
}

# Create and activate virtual environment
Write-Host "Setting up Python virtual environment..." -ForegroundColor Cyan
if (-Not (Test-Path -Path "venv")) {
    python -m venv venv
    Write-Host "  Created virtual environment."
}
else {
    Write-Host "  Virtual environment already exists."
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Cyan
& .\venv\Scripts\Activate.ps1

# Install dependencies
Write-Host "Installing dependencies..." -ForegroundColor Cyan
if (Test-Path -Path "requirements.txt") {
    pip install -r requirements.txt
    Write-Host "  Dependencies installed successfully."
}
else {
    Write-Host "  Warning: requirements.txt not found. Installing minimal dependencies..."
    pip install opencv-python numpy
}

# Project setup complete
Write-Host "`nProject setup complete!" -ForegroundColor Green
Write-Host "You can align images by running: python align_images.py" -ForegroundColor Yellow
Write-Host "Make sure to place your source images in the 'img' folder." -ForegroundColor Yellow
