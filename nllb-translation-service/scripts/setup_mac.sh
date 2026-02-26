#!/bin/bash
# MacBook Setup Script for NLLB Translation Service
# Supports both Apple Silicon (M1/M2/M3) and Intel Macs

set -e  # Exit on error

echo "=================================="
echo "NLLB Translation Service Setup"
echo "MacBook Configuration"
echo "=================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Detect architecture
ARCH=$(uname -m)
if [ "$ARCH" = "arm64" ]; then
    echo -e "${GREEN}Detected: Apple Silicon (M1/M2/M3)${NC}"
    IS_APPLE_SILICON=true
else
    echo -e "${YELLOW}Detected: Intel Mac${NC}"
    IS_APPLE_SILICON=false
fi

echo ""

# Check Python version
echo "Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 not found. Please install Python 3.10+${NC}"
    echo "Install via Homebrew: brew install python@3.10"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d ' ' -f 2)
echo -e "${GREEN}Found Python $PYTHON_VERSION${NC}"

# Check if virtualenv is installed
if ! python3 -m pip list | grep -q virtualenv; then
    echo "Installing virtualenv..."
    python3 -m pip install virtualenv
fi

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo -e "${GREEN}Virtual environment created${NC}"
else
    echo -e "${YELLOW}Virtual environment already exists${NC}"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch based on architecture
echo ""
echo "Installing PyTorch..."
if [ "$IS_APPLE_SILICON" = true ]; then
    echo -e "${GREEN}Installing PyTorch with MPS (Metal Performance Shaders) support${NC}"
    pip install torch torchvision torchaudio
else
    echo -e "${YELLOW}Installing CPU-only PyTorch${NC}"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Install core requirements
echo ""
echo "Installing core requirements..."
pip install -r requirements.txt

# Ask if user wants training dependencies
echo ""
read -p "Install training dependencies (LoRA/PEFT)? [y/N]: " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Installing training dependencies..."
    pip install -r requirements-training.txt
    echo -e "${GREEN}Training dependencies installed${NC}"
fi

# Ask if user wants development dependencies
echo ""
read -p "Install development dependencies (testing, linting)? [y/N]: " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Installing development dependencies..."
    pip install -r requirements-dev.txt
    echo -e "${GREEN}Development dependencies installed${NC}"
fi

# Create necessary directories
echo ""
echo "Creating necessary directories..."
mkdir -p models/cache
mkdir -p models/custom-nllb
mkdir -p models/lora-adapters
mkdir -p logs
mkdir -p data/training
echo -e "${GREEN}Directories created${NC}"

# Copy .env.example to .env if not exists
if [ ! -f ".env" ]; then
    echo ""
    echo "Creating .env file from template..."
    cp .env.example .env

    # Update .env for Mac
    if [ "$IS_APPLE_SILICON" = true ]; then
        # Enable MPS for Apple Silicon
        echo "" >> .env
        echo "# Mac-specific settings (Apple Silicon)" >> .env
        echo "USE_GPU=false  # MPS will be used automatically" >> .env
        echo "USE_HALF_PRECISION=false  # MPS doesn't support FP16 yet" >> .env
    else
        # CPU-only for Intel
        echo "" >> .env
        echo "# Mac-specific settings (Intel)" >> .env
        echo "USE_GPU=false" >> .env
        echo "USE_HALF_PRECISION=false" >> .env
    fi

    echo -e "${GREEN}.env file created. Please review and update as needed.${NC}"
else
    echo -e "${YELLOW}.env file already exists${NC}"
fi

# Test installation
echo ""
echo "Testing installation..."
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python3 -c "import transformers; print(f'Transformers version: {transformers.__version__}')"

if [ "$IS_APPLE_SILICON" = true ]; then
    python3 -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
fi

echo ""
echo -e "${GREEN}=================================="
echo "Setup Complete!"
echo -e "==================================${NC}"
echo ""
echo "Next steps:"
echo "1. Activate the virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "2. Download the model (optional, will download on first use):"
echo "   python scripts/download_model.py"
echo ""
echo "3. Start the development server:"
echo "   python -m app.main"
echo ""
echo "4. Or use uvicorn directly:"
echo "   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"
echo ""
echo "5. Visit http://localhost:8000/docs for API documentation"
echo ""
echo -e "${YELLOW}Note: For Apple Silicon Macs:${NC}"
echo "  - MPS (Metal Performance Shaders) will be used for acceleration"
echo "  - First translation may be slow as the model compiles for MPS"
echo "  - FP16 is not yet supported on MPS, using FP32"
echo ""
