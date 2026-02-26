# macOS Setup Guide for NLLB Translation Service

Complete guide for setting up and running the NLLB Translation Service on macOS (both Intel and Apple Silicon).

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Quick Start](#quick-start)
3. [Manual Setup](#manual-setup)
4. [Apple Silicon (M1/M2/M3) Optimization](#apple-silicon-optimization)
5. [Intel Mac Setup](#intel-mac-setup)
6. [Running the Service](#running-the-service)
7. [Training on Mac](#training-on-mac)
8. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required Software
- **macOS**: 11.0 (Big Sur) or later
- **Python**: 3.10 or later
- **Homebrew**: Package manager for macOS
- **Git**: Version control
- **8GB+ RAM**: Minimum (16GB+ recommended)
- **20GB+ free disk space**: For models and dependencies

### Install Homebrew (if not installed)
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### Install Python
```bash
# Install Python 3.10
brew install python@3.10

# Verify installation
python3 --version
```

---

## Quick Start

The fastest way to get started on macOS:

```bash
# 1. Clone the repository
git clone <repository-url>
cd nllb-translation-service

# 2. Run the automated setup script
chmod +x scripts/setup_mac.sh
./scripts/setup_mac.sh

# 3. Activate virtual environment
source venv/bin/activate

# 4. Start the service
python -m app.main
```

The setup script will:
- Detect your Mac architecture (Intel vs Apple Silicon)
- Create a virtual environment
- Install appropriate PyTorch version
- Install all dependencies
- Configure environment variables
- Create necessary directories

---

## Manual Setup

If you prefer manual setup or want more control:

### 1. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
```

### 2. Install PyTorch

#### For Apple Silicon (M1/M2/M3)
```bash
# Install PyTorch with MPS support
pip install torch torchvision torchaudio
```

#### For Intel Mac
```bash
# Install CPU-only PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 3. Install Dependencies
```bash
# Core dependencies
pip install -r requirements.txt

# Training dependencies (optional)
pip install -r requirements-training.txt

# Development dependencies (optional)
pip install -r requirements-dev.txt
```

### 4. Configure Environment
```bash
# Copy environment template
cp .env.example .env

# Edit .env file
nano .env  # or use your preferred editor
```

#### For Apple Silicon
```env
USE_GPU=false  # MPS will be auto-detected
USE_HALF_PRECISION=false  # MPS doesn't support FP16 yet
```

#### For Intel Mac
```env
USE_GPU=false
USE_HALF_PRECISION=false
```

### 5. Create Directories
```bash
mkdir -p models/cache models/custom-nllb models/lora-adapters logs data/training
```

---

## Apple Silicon Optimization

### MPS (Metal Performance Shaders)
Apple Silicon Macs (M1/M2/M3) use MPS for GPU acceleration:

**Advantages:**
- ✅ Native GPU acceleration
- ✅ Unified memory architecture
- ✅ Energy efficient
- ✅ 2-3x faster than CPU

**Limitations:**
- ⚠️ FP16 not yet fully supported (use FP32)
- ⚠️ First inference is slow (compilation)
- ⚠️ Some operations fall back to CPU

### Performance Tips for Apple Silicon
```python
# In your config or code
import torch

# Check MPS availability
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple Silicon GPU (MPS)")
else:
    device = torch.device("cpu")
```

### Expected Performance (Apple Silicon)
| Model Size | First Translation | Subsequent | Throughput |
|-----------|------------------|-----------|-----------|
| NLLB-600M | 5-10s (compile) | 200-500ms | 2-5 req/s |
| NLLB-1.3B | 10-15s (compile) | 400-800ms | 1-3 req/s |
| NLLB-3.3B | 20-30s (compile) | 800-1500ms | 0.5-1 req/s |

---

## Intel Mac Setup

Intel Macs will use CPU-only inference:

### Performance Optimization
```bash
# Install optimized libraries
pip install intel-extension-for-pytorch  # Optional, for newer Intel chips

# Use smaller batch sizes
# In .env file:
TRAINING_BATCH_SIZE=4
MAX_LENGTH=256  # Reduce if memory constrained
```

### Expected Performance (Intel Mac)
| Model Size | Translation Time | Throughput |
|-----------|-----------------|-----------|
| NLLB-600M | 500-1000ms | 1-2 req/s |
| NLLB-1.3B | 1000-2000ms | 0.5-1 req/s |

**Tip**: For production use on Intel Mac, consider using ONNX Runtime for better performance.

---

## Running the Service

### Development Mode
```bash
# Activate environment
source venv/bin/activate

# Run with auto-reload
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Production Mode
```bash
# Using gunicorn
gunicorn app.main:app \
  --workers 2 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --timeout 120
```

### Access Points
- **API Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/api/v1/health

### Test Translation
```bash
curl -X POST "http://localhost:8000/api/v1/translate" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, how are you?",
    "source_lang": "eng_Latn",
    "target_lang": "hin_Deva"
  }'
```

---

## Training on Mac

### LoRA Training (Recommended for Mac)
LoRA is highly recommended for Mac due to lower memory requirements:

```bash
# Prepare training data (TSV format: source<TAB>target)
echo -e "Hello\tनमस्ते" > data/training/sample.tsv
echo -e "Thank you\tधन्यवाद" >> data/training/sample.tsv

# Train with LoRA
python training/train_lora.py \
  --data-file data/training/sample.tsv \
  --source-lang eng_Latn \
  --target-lang hin_Deva \
  --output-dir models/lora-adapters/my-model \
  --config configs/training_config.yaml \
  --lora-r 8 \
  --lora-alpha 16 \
  --batch-size 4 \
  --epochs 3
```

### Memory Requirements
| Training Type | Memory Required | Batch Size |
|--------------|----------------|-----------|
| LoRA (r=8) | 6-8 GB | 4 |
| LoRA (r=16) | 8-10 GB | 2-4 |
| Full Fine-tune | 16+ GB | 1-2 |

### Apple Silicon Training Tips
```yaml
# In configs/training_config.yaml
training:
  hyperparameters:
    per_device_train_batch_size: 4  # Adjust based on RAM
    gradient_accumulation_steps: 4  # Simulate larger batch
    fp16: false  # Use FP32 for MPS

lora:
  r: 8  # Lower rank = less memory
  lora_alpha: 16
```

---

## Troubleshooting

### Common Issues and Solutions

#### 1. PyTorch Not Finding MPS
```bash
# Check MPS availability
python3 -c "import torch; print(torch.backends.mps.is_available())"

# Update PyTorch to latest
pip install --upgrade torch torchvision torchaudio
```

#### 2. Out of Memory Errors
```bash
# Reduce batch size in .env
MAX_LENGTH=256
TRAINING_BATCH_SIZE=2

# Or use CPU instead
USE_GPU=false
```

#### 3. Slow First Translation (Apple Silicon)
This is normal! MPS compiles the model on first use.
```python
# Add warmup in app/main.py lifespan
from inference.warmup import warm_up_model
warm_up_model(model, tokenizer, device="mps")
```

#### 4. "Symbol not found" Errors
```bash
# Reinstall PyTorch
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio
```

#### 5. Port Already in Use
```bash
# Find and kill process
lsof -ti:8000 | xargs kill -9

# Or use different port
uvicorn app.main:app --port 8001
```

#### 6. Model Download Fails
```bash
# Manually download model
python scripts/download_model.py

# Or set HF_HOME environment
export HF_HOME=~/huggingface
```

### Performance Debugging
```bash
# Monitor CPU/Memory usage
top

# Monitor GPU (Apple Silicon)
sudo powermetrics --samplers gpu_power -i 1000 -n 1

# Profile your code
python -m cProfile -o profile.stats app/main.py
```

---

## Docker on Mac

### Install Docker Desktop
```bash
brew install --cask docker
```

### Build and Run
```bash
# Build image (CPU-only)
docker build -t nllb-translation:latest .

# Run container
docker run -p 8000:8000 \
  -e USE_GPU=false \
  -v $(pwd)/models:/app/models \
  nllb-translation:latest
```

**Note**: Docker on Mac doesn't support GPU acceleration (no MPS in containers).

---

## Production Deployment on Mac

For running in production on a Mac server:

### 1. Use Process Manager
```bash
# Install pm2
brew install node
npm install -g pm2

# Start service
pm2 start app/main.py --interpreter python3 --name nllb-translation

# Monitor
pm2 monit

# Enable startup
pm2 startup
pm2 save
```

### 2. Setup Nginx Reverse Proxy
```bash
# Install nginx
brew install nginx

# Configure (see docs/DEPLOYMENT.md)
sudo nano /usr/local/etc/nginx/nginx.conf

# Start nginx
brew services start nginx
```

### 3. Enable HTTPS
```bash
# Install certbot
brew install certbot

# Generate certificate
sudo certbot --nginx -d yourdomain.com
```

---

## Performance Benchmarks

### Apple Silicon (M1 Pro, 16GB)
| Task | Time | Notes |
|------|------|-------|
| Model Loading | 8-12s | First time only |
| Warmup | 5-8s | MPS compilation |
| Translation (short) | 200-300ms | <50 words |
| Translation (medium) | 400-600ms | 50-150 words |
| Translation (long) | 800-1200ms | 150-500 words |
| Batch (10 texts) | 2-3s | Parallel processing |

### Intel Mac (i7, 16GB)
| Task | Time | Notes |
|------|------|-------|
| Model Loading | 10-15s | Slower than M1 |
| Translation (short) | 500-800ms | CPU-only |
| Translation (medium) | 1-1.5s | CPU-only |
| Translation (long) | 2-3s | CPU-only |

---

## Next Steps

- ✅ [Training Guide](TRAINING.md) - Fine-tune for your domain
- ✅ [API Documentation](API.md) - Complete API reference
- ✅ [Deployment Guide](DEPLOYMENT.md) - Production deployment
- ✅ [Contributing](../CONTRIBUTING.md) - Contribute to the project

---

## Support

For Mac-specific issues:
- Check Apple Silicon compatibility: https://github.com/pytorch/pytorch/issues
- PyTorch MPS documentation: https://pytorch.org/docs/stable/notes/mps.html

For general issues:
- Open an issue on GitHub
- Check existing documentation
- Join our community discussions
