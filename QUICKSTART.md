# Quick Start Guide

Get the NLLB Translation Service running in 5 minutes.

## Prerequisites

- Python 3.10+
- Docker and Docker Compose (for containerized deployment)
- 4GB+ RAM (8GB+ recommended)
- Optional: NVIDIA GPU with CUDA support

## Installation Steps

### Option 1: Docker (Fastest)

```bash
# 1. Copy environment configuration
cp .env.example .env

# 2. Start the service
docker-compose up -d

# 3. Check status
docker-compose logs -f

# 4. Test the API
curl http://localhost:8000/api/v1/health
```

**Access the service:**
- API: http://localhost:8000
- Documentation: http://localhost:8000/docs

### Option 2: Local Python

```bash
# 1. Create virtual environment
python3.10 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Copy environment file
cp .env.example .env

# 4. Run the service
python -m app.main
```

## First Translation

### Using curl:

```bash
curl -X POST "http://localhost:8000/api/v1/translate" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, how are you?",
    "source_lang": "eng_Latn",
    "target_lang": "hin_Deva"
  }'
```

### Using Python:

```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/translate",
    json={
        "text": "Hello, how are you?",
        "source_lang": "eng_Latn",
        "target_lang": "hin_Deva"
    }
)

print(response.json())
# Output: {'translated_text': 'नमस्ते, आप कैसे हैं?', ...}
```

### Using the test script:

```bash
bash scripts/test_api.sh
```

## Common Language Codes

| Language | Code |
|----------|------|
| English | `eng_Latn` |
| Hindi | `hin_Deva` |
| Spanish | `spa_Latn` |
| French | `fra_Latn` |
| German | `deu_Latn` |
| Chinese (Simplified) | `zho_Hans` |
| Arabic | `arb_Arab` |
| Japanese | `jpn_Jpan` |

Full list: http://localhost:8000/api/v1/languages

## Next Steps

1. **Explore the API**: Visit http://localhost:8000/docs
2. **Try with glossary**: See examples in README.md
3. **Fine-tune for your domain**: Follow training guide in README.md
4. **Production deployment**: Configure environment variables

## Troubleshooting

### Port already in use
```bash
# Change port in .env
PORT=8001

# Or in docker-compose.yml
ports:
  - "8001:8000"
```

### Model download taking long
- First startup downloads ~600MB model
- Subsequent starts are fast (model cached)
- Pre-download: `python scripts/download_model.py`

### Out of memory
```bash
# In .env, set:
USE_HALF_PRECISION=true
MAX_LENGTH=256
```

## Support

- Full documentation: [README.md](README.md)
- API docs: http://localhost:8000/docs
- Issues: GitHub Issues

---

Happy translating! 🌍
