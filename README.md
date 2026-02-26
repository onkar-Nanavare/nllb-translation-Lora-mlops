<<<<<<< HEAD
# nllb-translation-Lora-mlops
=======
# NLLB Translation Service

A production-ready AI Translation service using Meta's NLLB (No Language Left Behind) model. This service provides enterprise-grade translation capabilities with support for 200+ languages, custom domain fine-tuning, glossary management, and optimized inference.

## Features

- **200+ Languages**: Support for Meta's NLLB-200 model covering 200+ languages
- **REST API**: FastAPI-based REST endpoints with automatic documentation
- **Custom Training**: Fine-tune on domain-specific data (medical, legal, technical, etc.)
- **Glossary Support**: Term-level translation overrides for domain terminology
- **GPU Acceleration**: Automatic GPU detection with FP16 support
- **Production Ready**: Docker deployment, logging, rate limiting, error handling
- **Batch Processing**: Translate multiple texts efficiently
- **ONNX Export**: Convert models to ONNX for optimized inference
- **Comprehensive Testing**: Unit tests with pytest

## Architecture

```
nllb-translation-service/
├── app/
│   ├── main.py              # FastAPI application
│   ├── api/                 # API routes and middleware
│   ├── services/            # Translation and language services
│   ├── models/              # Pydantic schemas
│   ├── core/                # Configuration and logging
│   └── glossary/            # Glossary processing
├── training/
│   ├── train.py             # Fine-tuning script
│   ├── evaluate.py          # Evaluation script
│   └── export_onnx.py       # ONNX export
├── tests/                   # Unit tests
├── models/                  # Model storage
├── data/                    # Training data
├── Dockerfile               # Production Docker image
├── docker-compose.yml       # Container orchestration
└── requirements.txt         # Python dependencies
```

## Quick Start

### 🚀 Fastest Way (Automated Scripts)

**First time setup:**
```bash
./setup.sh
```

**Run the service:**
```bash
./run.sh
```

Access the service at:
- 🏠 Home: http://localhost:8000
- 📚 API Docs: http://localhost:8000/docs
- 💚 Health: http://localhost:8000/api/v1/health

---

### Option 1: Docker (Recommended for Production)

1. **Clone and setup**
```bash
git clone <repository-url>
cd nllb-translation-service

# Copy environment file
cp .env.example .env
```

2. **Start service**
```bash
docker-compose up -d
```

3. **Verify health**
```bash
curl http://localhost:8000/api/v1/health
```

### Option 2: Local Development (Manual)

1. **Install dependencies**
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Upgrade pip and install build tools
pip install --upgrade pip setuptools wheel cmake ninja

# Install PyTorch first
pip install torch>=2.10.0

# Install requirements
pip install -r requirements.txt
```

2. **Configure environment**
```bash
cp .env.example .env
# Edit .env as needed
```

3. **Run service**
```bash
# Development mode
python -m app.main

# Or with uvicorn
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## API Documentation

Once running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### API Endpoints

#### 1. Translate Text

```bash
curl -X POST "http://localhost:8000/api/v1/translate" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, how are you?",
    "source_lang": "eng_Latn",
    "target_lang": "hin_Deva"
  }'
```

**Response:**
```json
{
  "translated_text": "नमस्ते, आप कैसे हैं?",
  "source_lang": "eng_Latn",
  "target_lang": "hin_Deva",
  "glossary_applied": false
}
```

#### 2. Translate with Glossary

```bash
curl -X POST "http://localhost:8000/api/v1/translate" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Patient has hypertension and needs medication",
    "source_lang": "eng_Latn",
    "target_lang": "hin_Deva",
    "glossary": {
      "hypertension": "उच्च रक्तचाप",
      "medication": "दवा"
    }
  }'
```

#### 3. Batch Translation

```bash
curl -X POST "http://localhost:8000/api/v1/batch-translate" \
  -H "Content-Type: application/json" \
  -d '{
    "items": [
      {
        "text": "Hello",
        "source_lang": "eng_Latn",
        "target_lang": "hin_Deva"
      },
      {
        "text": "Good morning",
        "source_lang": "eng_Latn",
        "target_lang": "spa_Latn"
      }
    ]
  }'
```

#### 4. Get Supported Languages

```bash
curl http://localhost:8000/api/v1/languages
```

#### 5. Health Check

```bash
curl http://localhost:8000/api/v1/health
```

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "model_loaded": true,
  "gpu_available": true
}
```

## Supported Languages

The service supports 200+ language codes. Common examples:

| Code | Language |
|------|----------|
| `eng_Latn` | English |
| `hin_Deva` | Hindi |
| `spa_Latn` | Spanish |
| `fra_Latn` | French |
| `deu_Latn` | German |
| `zho_Hans` | Chinese (Simplified) |
| `zho_Hant` | Chinese (Traditional) |
| `ara_Arab` | Modern Standard Arabic |
| `jpn_Jpan` | Japanese |
| `kor_Hang` | Korean |
| `por_Latn` | Portuguese |
| `rus_Cyrl` | Russian |
| `urd_Arab` | Urdu |
| `ben_Beng` | Bengali |
| `vie_Latn` | Vietnamese |

**See full list**: `GET /api/v1/languages`

## Custom Domain Training

### 1. Prepare Training Data

Create a TSV file with source and target text:

```tsv
# data/medical_en_hi.tsv
The patient has hypertension	रोगी को उच्च रक्तचाप है
Please take this medication twice daily	कृपया यह दवा दिन में दो बार लें
The blood test results are normal	रक्त परीक्षण के परिणाम सामान्य हैं
```

### 2. Train Model

```bash
python training/train.py \
  --data-file data/medical_en_hi.tsv \
  --source-lang eng_Latn \
  --target-lang hin_Deva \
  --output-dir ./models/custom-nllb-medical \
  --epochs 3 \
  --batch-size 8 \
  --learning-rate 2e-5
```

**Training Options:**
- `--data-file`: Path to TSV training data
- `--source-lang`: Source language code
- `--target-lang`: Target language code
- `--output-dir`: Where to save fine-tuned model
- `--epochs`: Number of training epochs (default: 3)
- `--batch-size`: Training batch size (default: 8)
- `--learning-rate`: Learning rate (default: 2e-5)
- `--eval-split`: Validation split fraction (default: 0.1)

### 3. Evaluate Model

```bash
python training/evaluate.py \
  --model-path ./models/custom-nllb-medical \
  --test-file data/test_medical_en_hi.tsv \
  --source-lang eng_Latn \
  --target-lang hin_Deva
```

**Output:**
```
================================================================================
Evaluation Results
================================================================================
bleu: 45.23
char_accuracy: 78.50
avg_length_ratio: 0.95
================================================================================
```

### 4. Use Custom Model

Update `.env`:
```env
CUSTOM_MODEL_PATH=./models/custom-nllb-medical
```

Restart the service:
```bash
docker-compose restart
```

## ONNX Export for Optimized Inference

Convert your model to ONNX format:

```bash
python training/export_onnx.py \
  --model-path ./models/custom-nllb-medical \
  --output-path ./models/onnx-medical \
  --validate
```

**Benefits:**
- Faster inference
- Lower memory usage
- Better CPU performance

**To use ONNX model:**

1. Install ONNX runtime:
```bash
pip install optimum[onnxruntime]
# For GPU: pip install optimum[onnxruntime-gpu]
```

2. Update `.env`:
```env
CUSTOM_MODEL_PATH=./models/onnx-medical
```

## GPU Setup

### Docker with GPU

1. Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

2. Update `docker-compose.yml` to use the GPU service:
```bash
# Uncomment the translation-service-gpu section
docker-compose up translation-service-gpu -d
```

### Local GPU Setup

1. Install CUDA-compatible PyTorch:
```bash
pip install torch --extra-index-url https://download.pytorch.org/whl/cu118
```

2. Update `.env`:
```env
USE_GPU=true
USE_HALF_PRECISION=true
```

## Production Deployment

### Environment Variables

Key production settings:

```env
# Performance
NUM_WORKERS=4
RATE_LIMIT_PER_MINUTE=100
REQUEST_TIMEOUT=60

# Security
DEBUG=false

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json

# Model
USE_GPU=true
USE_HALF_PRECISION=true
```

### Scaling

**Horizontal Scaling:**
```bash
docker-compose up --scale translation-service=3
```

**Load Balancer:**
Use Nginx or Traefik to distribute requests.

### Monitoring

Logs are structured JSON format:
```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "level": "INFO",
  "message": "Translation completed",
  "request_id": "abc-123",
  "duration": 0.245
}
```

## Testing

Run tests:
```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run all tests
pytest

# With coverage
pytest --cov=app --cov-report=html

# Run specific test file
pytest tests/test_api.py

# Run with verbose output
pytest -v
```

## Development

### Code Quality

```bash
# Format code
black app/ tests/

# Sort imports
isort app/ tests/

# Lint
flake8 app/ tests/

# Type check
mypy app/
```

### Pre-commit Hooks

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

## Performance Optimization

### Tips for Best Performance:

1. **GPU**: Use GPU with FP16 for 2-3x speedup
2. **Batch Processing**: Use batch endpoint for multiple translations
3. **ONNX**: Export to ONNX for CPU optimization
4. **Caching**: Model files cached locally after first download
5. **Workers**: Tune `NUM_WORKERS` based on CPU cores

### Benchmarks (approximate):

| Setup | Throughput | Latency |
|-------|------------|---------|
| CPU (base) | ~10 req/s | ~100ms |
| GPU (FP32) | ~50 req/s | ~20ms |
| GPU (FP16) | ~100 req/s | ~10ms |
| ONNX (CPU) | ~30 req/s | ~30ms |

## Troubleshooting

### Model Download Issues
```bash
# Manually download model
export HF_HOME=./models/cache
python -c "from transformers import AutoTokenizer, AutoModelForSeq2SeqLM; \
           AutoTokenizer.from_pretrained('facebook/nllb-200-distilled-600M'); \
           AutoModelForSeq2SeqLM.from_pretrained('facebook/nllb-200-distilled-600M')"
```

### Out of Memory
- Reduce `MAX_LENGTH` in `.env`
- Use smaller batch size for training
- Enable `USE_HALF_PRECISION=true`
- Reduce `NUM_WORKERS`

### Slow First Request
- Model loads on startup (can take 30-60s)
- Subsequent requests are fast
- Use health check with `start_period` in Docker

## License

This project is licensed under the MIT License.

## Acknowledgments

- Meta AI for the NLLB model
- HuggingFace for Transformers library
- FastAPI for the web framework

## Support

For issues and questions:
- GitHub Issues: [Create an issue]
- Documentation: See `/docs` endpoint when running

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request

---

**Built with ❤️ using Meta NLLB-200 and FastAPI**
>>>>>>> 2d43645 (Initial commit with data, workflows, and project files)
