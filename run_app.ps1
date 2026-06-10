# run_app.ps1 — Activate the virtual environment and launch the Streamlit app.

$venvActivate = Join-Path $PSScriptRoot ".venv\Scripts\Activate.ps1"

if (Test-Path $venvActivate) {
    Write-Host "Activating virtual environment..." -ForegroundColor Cyan
    & $venvActivate
} else {
    Write-Warning ".venv not found at $venvActivate — running with system Python."
}

Write-Host "Starting PolicyMind AI Streamlit app..." -ForegroundColor Green
streamlit run app/streamlit_app.py
