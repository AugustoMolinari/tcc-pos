#!/usr/bin/env bash
# Book RAG Launcher Script
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

# Check if .venv exists
if [ ! -d ".venv" ]; then
    echo "Virtual environment not found. Creating one with uv..."
    if command -v uv &> /dev/null; then
        uv venv
        uv pip install -e .
    else
        python3 -m venv .venv
        .venv/bin/pip install -e .
    fi
fi

# Ensure Ollama daemon is running
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "Starting local Ollama server in background..."
    if command -v ollama &> /dev/null; then
        ollama serve > /dev/null 2>&1 &
    elif [ -f "$HOME/.local/bin/ollama" ]; then
        "$HOME/.local/bin/ollama" serve > /dev/null 2>&1 &
    else
        echo "Warning: Ollama is not installed or not in PATH."
        echo "Install Ollama via: curl -fsSL https://ollama.com/install.sh | sh"
    fi
    sleep 2
fi

# Run Book RAG
exec "$DIR/.venv/bin/python" -m src.cli "$@"
