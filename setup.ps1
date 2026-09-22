# ============================================================
# Finance AI - Full System Launch Script (Windows Local)
#
# • Ollama runs natively on your Windows PC (using your RTX 4060 GPU)
# • Postgres, ChromaDB, Signal Engine & API run in Docker
# ============================================================

Write-Host ""
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  Finance AI & Market Signal System - Local Launch" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host ""

# ── 1. Check Ollama ──────────────────────────────────────────
Write-Host "1. Checking Ollama..." -ForegroundColor Yellow
if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    Write-Host ""
    Write-Host "[!] Ollama is not found in PATH." -ForegroundColor Red
    Write-Host "Please install from: https://ollama.com/download/windows" -ForegroundColor White
    Write-Host "If installed, ensure Ollama is running in your system tray." -ForegroundColor White
    exit 1
}
Write-Host "  -> Ollama found." -ForegroundColor Green

# ── 2. Check Docker Desktop ──────────────────────────────────
Write-Host "2. Checking Docker Desktop..." -ForegroundColor Yellow
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Host ""
    Write-Host "[!] Docker Desktop is not installed." -ForegroundColor Red
    Write-Host "Please download from: https://www.docker.com/products/docker-desktop/" -ForegroundColor White
    exit 1
}

$dockerInfo = docker info 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "[!] Docker Desktop is installed but not running." -ForegroundColor Red
    Write-Host "Please open Docker Desktop and wait until the engine starts." -ForegroundColor White
    exit 1
}
Write-Host "  -> Docker Desktop is running." -ForegroundColor Green

# ── 3. Check / Create .env ───────────────────────────────────
Write-Host "3. Checking configuration..." -ForegroundColor Yellow
if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "  -> Created .env from template." -ForegroundColor Green
} else {
    Write-Host "  -> .env file exists." -ForegroundColor Green
}

# ── 4. Pull LLM & Embedding Models ───────────────────────────
Write-Host "4. Ensuring Ollama models are pulled..." -ForegroundColor Yellow
Write-Host "  Pulling llama3.1:8b (quantized for 8GB RTX 4060)..." -ForegroundColor Gray
ollama pull llama3.1:8b

Write-Host "  Pulling nomic-embed-text (for RAG vector search)..." -ForegroundColor Gray
ollama pull nomic-embed-text
Write-Host "  -> Local AI models ready." -ForegroundColor Green

# ── 5. Build & Start Docker Services ─────────────────────────
Write-Host ""
Write-Host "5. Launching Docker Microservices (Postgres, Signal Engine, API, Open WebUI)..." -ForegroundColor Yellow
docker compose up -d --build

if ($LASTEXITCODE -ne 0) {
    Write-Host "[!] Docker Compose failed to start. Check logs." -ForegroundColor Red
    exit 1
}

# ── 6. Ready! ────────────────────────────────────────────────
Write-Host ""
Write-Host "==========================================================" -ForegroundColor Green
Write-Host "  Finance AI & Market Advisory System is LIVE! 🚀" -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Green
Write-Host ""
Write-Host "  • Chat Interface (Open WebUI):   http://localhost:3000" -ForegroundColor Cyan
Write-Host "  • Finance & Trading API Docs:    http://localhost:8080/docs" -ForegroundColor Cyan
Write-Host "  • Live Market Signal Summary:    http://localhost:8080/market/summary" -ForegroundColor Cyan
Write-Host "  • PostgreSQL Port:               localhost:5432" -ForegroundColor Gray
Write-Host ""
Write-Host "Opening Chat UI in browser in 3 seconds..." -ForegroundColor Yellow
Start-Sleep -Seconds 3
Start-Process "http://localhost:3000"
