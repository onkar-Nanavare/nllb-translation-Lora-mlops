# Architecture Overview

## System Design

```
┌─────────────┐
│   Client    │
│  (Browser,  │
│   App, CLI) │
└──────┬──────┘
       │ HTTP/REST
       ▼
┌─────────────────────────────────────────┐
│         FastAPI Application             │
│  ┌───────────────────────────────────┐  │
│  │     Middleware Layer              │  │
│  │  • Request Logging                │  │
│  │  • Rate Limiting                  │  │
│  │  • Error Handling                 │  │
│  │  • CORS                           │  │
│  └───────────────────────────────────┘  │
│                                          │
│  ┌───────────────────────────────────┐  │
│  │     API Routes                    │  │
│  │  • /translate                     │  │
│  │  • /batch-translate               │  │
│  │  • /languages                     │  │
│  │  • /health                        │  │
│  └───────────────────────────────────┘  │
│                                          │
│  ┌───────────────────────────────────┐  │
│  │     Business Logic                │  │
│  │  • Translation Service            │  │
│  │  • Glossary Processor             │  │
│  │  • Language Validation            │  │
│  └───────────────────────────────────┘  │
└─────────────────────────────────────────┘
                   │
                   ▼
         ┌──────────────────┐
         │   NLLB Model     │
         │  (Transformers)  │
         │  • Tokenizer     │
         │  • Seq2Seq Model │
         └──────────────────┘
                   │
                   ▼
         ┌──────────────────┐
         │   GPU / CPU      │
         │   (PyTorch)      │
         └──────────────────┘
```

## Component Overview

### 1. API Layer (`app/api/`)

**Routes** (`routes.py`)
- Defines REST endpoints
- Request validation with Pydantic
- Response formatting
- Error handling

**Middleware** (`middleware.py`)
- `RequestLoggingMiddleware`: Logs all requests/responses
- `RateLimitMiddleware`: In-memory rate limiting
- `ErrorHandlingMiddleware`: Global exception handling

### 2. Core Services (`app/services/`)

**Translation Service** (`translator.py`)
- Singleton pattern for model management
- Model loading and caching
- GPU detection and optimization
- Translation execution with torch.no_grad()

**Language Service** (`language_codes.py`)
- NLLB-200 language code mappings
- Language validation
- Language name resolution

### 3. Glossary Processing (`app/glossary/`)

**Glossary Processor** (`processor.py`)
- Post-processing layer
- Term-level overrides
- Case-sensitive/insensitive matching
- Multiple glossary merging

### 4. Data Models (`app/models/`)

**Pydantic Schemas** (`schemas.py`)
- Request/response validation
- Type safety
- API documentation generation
- Example data for OpenAPI

### 5. Configuration (`app/core/`)

**Settings** (`config.py`)
- Environment variable management
- Pydantic Settings integration
- Type-safe configuration
- Defaults and validation

**Logging** (`logging_config.py`)
- Structured JSON logging
- Multiple log levels
- Request ID tracking
- Performance metrics

## Data Flow

### Translation Request Flow

```
1. Client Request
   POST /api/v1/translate
   {
     "text": "Hello",
     "source_lang": "eng_Latn",
     "target_lang": "hin_Deva",
     "glossary": {...}
   }
   ▼

2. Middleware Processing
   • Log request (with request_id)
   • Check rate limit
   • Validate request
   ▼

3. API Route Handler
   • Parse JSON body
   • Validate Pydantic model
   • Extract parameters
   ▼

4. Translation Service
   • Validate language codes
   • Tokenize input text
   • Run model inference (GPU/CPU)
   • Decode output
   ▼

5. Glossary Processing (if applicable)
   • Match glossary terms
   • Apply overrides
   • Return modified text
   ▼

6. Response Formation
   • Create TranslationResponse
   • Add metadata
   • Return JSON
   ▼

7. Middleware Post-Processing
   • Log response
   • Add headers (request_id, rate_limit)
   • Return to client
```

## Model Architecture

### NLLB-200 Model

- **Type**: Sequence-to-Sequence Transformer
- **Architecture**: Encoder-Decoder
- **Size**: 600M parameters (distilled version)
- **Languages**: 200+ languages
- **Context**: Up to 512 tokens

### Inference Pipeline

```python
# Tokenization
inputs = tokenizer(text, return_tensors="pt")

# Encoding
encoder_outputs = model.encoder(inputs)

# Decoding with forced language token
decoder_outputs = model.decoder(
    encoder_outputs,
    forced_bos_token_id=target_lang_id
)

# Generation
output_ids = model.generate(
    inputs,
    forced_bos_token_id=target_lang_id,
    max_length=512,
    num_beams=5
)

# Detokenization
translated = tokenizer.decode(output_ids)
```

## Training Pipeline

### Fine-tuning Architecture

```
Training Data (TSV)
    ▼
Data Loading & Preprocessing
    ▼
Tokenization
    • Source text → input_ids
    • Target text → labels
    ▼
HuggingFace Trainer
    • Seq2SeqTrainingArguments
    • DataCollatorForSeq2Seq
    • Training loop
    ▼
Model Checkpoints
    ▼
Best Model Selection
    ▼
Save & Export
```

## Performance Optimizations

### 1. Model Level
- **GPU Acceleration**: CUDA support
- **Half Precision**: FP16 for 2x speedup
- **No Gradient**: torch.no_grad() for inference
- **Beam Search**: Quality vs speed tradeoff

### 2. Application Level
- **Model Caching**: Singleton pattern
- **Connection Pooling**: Reuse workers
- **Async Handlers**: FastAPI async support
- **Batch Processing**: Batch endpoint

### 3. Deployment Level
- **Multi-worker**: Gunicorn workers
- **Load Balancing**: Horizontal scaling
- **Docker Layers**: Efficient caching
- **ONNX Export**: Optimized runtime

## Security Considerations

### Input Validation
- Pydantic models for type safety
- Length limits on text inputs
- Language code validation
- Sanitization of user inputs

### Rate Limiting
- Per-client IP tracking
- Configurable limits
- 429 responses with Retry-After

### Error Handling
- Never expose internal errors
- Structured error responses
- Request ID tracking
- Comprehensive logging

### Production Hardening
- CORS configuration
- Environment-based secrets
- No debug mode in production
- Health check endpoints

## Scalability

### Horizontal Scaling
```bash
# Multiple instances
docker-compose up --scale translation-service=4
```

### Vertical Scaling
- GPU memory determines model size
- CPU cores determine worker count
- RAM affects batch sizes

### Caching Strategies
- Model files cached on disk
- In-memory rate limit cache
- Response caching (optional)

## Monitoring & Observability

### Structured Logging
```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "level": "INFO",
  "request_id": "abc-123",
  "duration": 0.245,
  "message": "Translation completed"
}
```

### Metrics to Track
- Request latency (p50, p95, p99)
- Throughput (requests/second)
- Error rates by type
- Model inference time
- GPU/CPU utilization

### Health Checks
- `/api/v1/health` endpoint
- Model loaded status
- GPU availability
- Service version

## Technology Stack

### Core Framework
- **FastAPI**: Modern async web framework
- **Pydantic**: Data validation
- **Uvicorn/Gunicorn**: ASGI server

### ML/AI
- **PyTorch**: Deep learning framework
- **Transformers**: HuggingFace library
- **NLLB**: Meta's translation model

### Infrastructure
- **Docker**: Containerization
- **Docker Compose**: Orchestration
- **CUDA**: GPU acceleration (optional)

### Development
- **pytest**: Testing framework
- **Black**: Code formatting
- **mypy**: Type checking

## Extension Points

### Adding New Features
1. Create new route in `app/api/routes.py`
2. Add Pydantic models in `app/models/schemas.py`
3. Implement business logic in `app/services/`
4. Add tests in `tests/`

### Custom Middleware
```python
# app/api/middleware.py
class CustomMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        # Custom logic
        response = await call_next(request)
        return response
```

### Model Swapping
- Change `MODEL_NAME` in `.env`
- Any HuggingFace Seq2Seq model compatible
- Adjust language codes if needed

---

For implementation details, see source code and README.md.
