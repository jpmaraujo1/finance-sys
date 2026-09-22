#!/bin/bash
# ============================================================
# Finance AI — Cloud GPU Bootstrap Script
# Run this on a fresh Ubuntu cloud GPU instance (RunPod, GCP)
# ============================================================

set -e

echo "🚀 Setting up Finance AI..."

# 1. Install Docker if not present
if ! command -v docker &> /dev/null; then
    echo "📦 Installing Docker..."
    curl -fsSL https://get.docker.com | sh
    sudo usermod -aG docker $USER
fi

# 2. Install Docker Compose
if ! command -v docker-compose &> /dev/null; then
    echo "📦 Installing Docker Compose..."
    sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" \
        -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
fi

# 3. Install NVIDIA Container Toolkit (for GPU support)
if ! dpkg -l | grep -q nvidia-container-toolkit; then
    echo "🎮 Installing NVIDIA Container Toolkit..."
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
    curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
        sudo tee /etc/apt/sources.list.d/nvidia-docker.list
    sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
    sudo systemctl restart docker
fi

# 4. Copy environment file
if [ ! -f .env ]; then
    cp .env.example .env
    echo "⚠️  .env file created. Edit it before continuing:"
    echo "    nano .env"
    echo "Press Enter when ready..."
    read
fi

# 5. Start all services
echo "🐳 Starting Docker services..."
docker-compose up -d

# 6. Wait for Ollama to be ready
echo "⏳ Waiting for Ollama to start..."
sleep 10

# 7. Pull the LLM model
echo "📥 Downloading LLaMA 3.1 8B model (this may take a few minutes)..."
docker exec finance-ai-ollama ollama pull llama3.1:8b

# 8. Pull the embedding model for RAG
echo "📥 Downloading embedding model..."
docker exec finance-ai-ollama ollama pull nomic-embed-text

echo ""
echo "✅ Finance AI is ready!"
echo ""
echo "🌐 Open WebUI (Chat):   http://localhost:3000"
echo "⚡ Finance API:          http://localhost:8080"
echo "🔍 ChromaDB:             http://localhost:8000"
echo "🤖 Ollama API:           http://localhost:11434"
