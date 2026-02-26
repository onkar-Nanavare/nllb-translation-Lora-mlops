#!/bin/bash

# NLLB Translation Service Setup Script

set -e  # Exit on error

echo "🚀 Setting up NLLB Translation Service..."
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip, setuptools, and wheel
echo "⬆️  Upgrading pip, setuptools, and wheel..."
pip install --upgrade pip setuptools wheel

# Install build dependencies first
echo "🔨 Installing build dependencies..."
pip install --upgrade cmake ninja

# Install PyTorch first (it's a large dependency and others depend on it)
echo "🔥 Installing PyTorch..."
pip install torch>=2.10.0

# Install the rest of the requirements
echo "📚 Installing remaining dependencies..."
pip install -r requirements.txt

echo ""
echo "✅ Setup complete!"
echo ""
echo "To activate the virtual environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To start the service, run:"
echo "  python -m uvicorn app.main:app --reload"
