# 🚀 Quick Start Guide - MacBook (5 Minutes)

Get your NLLB Translation Service running on macOS in under 5 minutes.

---

## Prerequisites Check

- ✅ macOS 11.0+ (Big Sur or later)
- ✅ 8GB+ RAM (16GB recommended)
- ✅ 20GB+ free disk space

---

## Step 1: Install Python (if needed)

```bash
# Check if Python 3.10+ is installed
python3 --version

# If not installed or version < 3.10:
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install python@3.10
```

---

## Step 2: Clone and Setup

```bash
# Navigate to your projects directory
cd ~/git/AI/v2

# If already cloned, cd into directory:
cd nllb-translation-service

# Run automated setup (detects Apple Silicon vs Intel automatically)
chmod +x scripts/setup_mac.sh
./scripts/setup_mac.sh
```

The setup script will:
- ✅ Create virtual environment
- ✅ Install PyTorch (MPS for M1/M2/M3, CPU for Intel)
- ✅ Install all dependencies
- ✅ Create necessary directories
- ✅ Configure .env file

**Time: ~2-3 minutes**

---

## Step 3: Activate Environment

```bash
source venv/bin/activate
```

You should see `(venv)` in your terminal prompt.

---

## Step 4: Start the Service

```bash
# Development mode (with auto-reload)
python -m app.main

# Or using uvicorn directly:
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**First Start**: Model will download automatically (~1.5GB). This takes 2-3 minutes.

**Subsequent Starts**: ~10-15 seconds

---

## Step 5: Test It! 🎉

### Option A: Use API Docs (Browser)

1. Open: http://localhost:8000/docs
2. Click on `/api/v1/translate`
3. Click "Try it out"
4. Enter:
   ```json
   {
     "text": "Hello, how are you?",
     "source_lang": "eng_Latn",
     "target_lang": "hin_Deva"
   }
   ```
5. Click "Execute"

### Option B: Use curl (Terminal)

```bash
curl -X POST "http://localhost:8000/api/v1/translate" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, how are you?",
    "source_lang": "eng_Latn",
    "target_lang": "hin_Deva"
  }'
```

**Expected Response:**
```json
{
  "translated_text": "नमस्ते, आप कैसे हैं?",
  "source_lang": "eng_Latn",
  "target_lang": "hin_Deva",
  "glossary_applied": false
}
```

---

## ⚡ Performance Notes

### Apple Silicon (M1/M2/M3)
- **First Translation**: 5-10 seconds (MPS compilation)
- **Subsequent Translations**: 200-500ms
- **Device**: MPS (Metal Performance Shaders) - GPU accelerated!

### Intel Mac
- **Translation**: 500-1000ms
- **Device**: CPU

**Check device:**
```bash
curl http://localhost:8000/api/v1/health | json_pp
```

---

## 🎯 What's Next?

### 1. Try Batch Translation
```bash
curl -X POST "http://localhost:8000/api/v1/batch-translate" \
  -H "Content-Type: application/json" \
  -d '{
    "items": [
      {"text": "Hello", "source_lang": "eng_Latn", "target_lang": "hin_Deva"},
      {"text": "World", "source_lang": "eng_Latn", "target_lang": "fra_Latn"},
      {"text": "Thank you", "source_lang": "eng_Latn", "target_lang": "spa_Latn"}
    ]
  }'
```

### 2. View Supported Languages
```bash
curl http://localhost:8000/api/v1/languages | json_pp
```

### 3. Check Service Health
```bash
curl http://localhost:8000/api/v1/health | json_pp
```

### 4. Fine-tune for Your Domain (Optional)
```bash
# Create training data (source<TAB>target)
echo -e "Medical term\tचिकित्सा शब्द" > data/training/medical.tsv
echo -e "Patient care\tरोगी देखभाल" >> data/training/medical.tsv


# Train LoRA adapter (fast and memory-efficient!)
python training/train_lora.py \
  --data-file data/training/medical.tsv \
  --source-lang eng_Latn \
  --target-lang hin_Deva \
  --output-dir models/lora-adapters/medical \
  --epochs 3 \
  --batch-size 4

# Update .env to use your adapter
echo "CUSTOM_MODEL_PATH=models/lora-adapters/medical" >> .env

# Restart service
```

---

## 🐛 Troubleshooting

### Port Already in Use
```bash
# Kill process on port 8000
lsof -ti:8000 | xargs kill -9

# Or use different port
uvicorn app.main:app --port 8001
```

### Out of Memory (Intel Mac)
```bash
# Edit .env file
nano .env

# Add these lines:
MAX_LENGTH=256
TRAINING_BATCH_SIZE=2
```

### Slow Performance (Apple Silicon)
First translation is slow (MPS compilation). This is normal!
```bash
# Add warmup to speed up first request
# Already configured in app/main.py ✅
```

### Model Download Fails
```bash
# Manually download
python scripts/download_model.py

# Or set custom cache dir
export HF_HOME=~/huggingface
```

---

## 📊 Monitor Your Service

### View Metrics
```bash
curl http://localhost:8000/metrics
```

### Setup Grafana (Optional)
```bash
# Install Docker Desktop first
brew install --cask docker

# Start monitoring stack
docker-compose -f docker-compose.yml up -d prometheus grafana

# Access:
# Prometheus: http://localhost:9090
# Grafana: http://localhost:3000 (admin/admin)
```

---

## 🎓 Learn More

- **Full Setup Guide**: `docs/MACOS_SETUP.md`
- **API Documentation**: http://localhost:8000/docs
- **Architecture**: `PRODUCTION_ARCHITECTURE.md`
- **Training Guide**: `configs/training_config.yaml`
- **All Features**: `REFACTORING_SUMMARY.md`

---

## 💡 Pro Tips

1. **First translation is slow** - MPS compilation (Apple Silicon). Subsequent are fast!
2. **Use batch API** - 3-5x faster for multiple texts
3. **Fine-tune with LoRA** - Only 10-100MB adapter files vs 1GB+ full models
4. **Monitor metrics** - `/metrics` endpoint for Prometheus
5. **Check health** - `/api/v1/health` for status

---

## 🎉 You're Ready!

Your translation service is now running on your Mac!

**API Endpoints:**
- 📖 Docs: http://localhost:8000/docs
- 🏥 Health: http://localhost:8000/api/v1/health
- 🔄 Translate: http://localhost:8000/api/v1/translate
- 📦 Batch: http://localhost:8000/api/v1/batch-translate
- 🌍 Languages: http://localhost:8000/api/v1/languages
- 📊 Metrics: http://localhost:8000/metrics

---

## ⏱️ Total Time: ~5 Minutes

- Setup: 2-3 min
- Model Download: 2-3 min (first time only)
- Start Service: 10-15 sec

**Happy Translating! 🌍**

---

## 🆘 Need Help?

```bash
# Rerun setup if something went wrong
./scripts/setup_mac.sh

# Check logs
tail -f logs/app.log

# Restart service
# Press Ctrl+C to stop, then:
python -m app.main
```

For more help, check `docs/MACOS_SETUP.md` or open an issue on GitHub.
