# Fixes Applied - NLLB Translation Service

## Issue Summary
The service was encountering a `TokenizersBackend has no attribute lang_code_to_id` error due to API changes in newer versions of the transformers library.

## Root Cause
The code was using the old transformers API (`tokenizer.lang_code_to_id[lang]`) which was changed in transformers v5.x. The newer version uses a different method to get language token IDs.

## Files Fixed

### 1. [app/services/translator.py](app/services/translator.py)
**Issue**: Line 159 - `self.tokenizer.lang_code_to_id[target_lang]`

**Fix**: Added compatibility layer to support both old and new tokenizer APIs:
```python
# Get target language token ID
# Handle both old and new tokenizer API
if hasattr(self.tokenizer, 'lang_code_to_id'):
    target_lang_id = self.tokenizer.lang_code_to_id[target_lang]
else:
    # For newer transformers versions, convert token to ID
    target_lang_id = self.tokenizer.convert_tokens_to_ids(target_lang)
```

### 2. [inference/warmup.py](inference/warmup.py)
**Issue**: Lines 80, 103 - `tokenizer.lang_code_to_id[target_lang]`

**Fix**: Applied same compatibility fix for all model warmup operations.

### 3. [inference/batch_processor.py](inference/batch_processor.py)
**Issue**: Line 89 - `self.tokenizer.lang_code_to_id[target_lang]`

**Fix**: Applied compatibility fix for batch processing operations.

### 4. [training/evaluate.py](training/evaluate.py)
**Issue**: Line 104 - `self.tokenizer.lang_code_to_id[self.target_lang]`

**Fix**: Applied compatibility fix for model evaluation script.

### 5. [training/export_onnx.py](training/export_onnx.py)
**Issue**: Line 114 - `tokenizer.lang_code_to_id[target_lang]`

**Fix**: Applied compatibility fix for ONNX export validation.

## Dependency Updates

### [requirements.txt](requirements.txt)
Updated all dependencies to versions compatible with Python 3.13+:

- ✅ `torch`: 2.1.2 → 2.10.0 (fixes availability issue)
- ✅ `transformers`: 4.36.2 → 4.48.0+ (newer API)
- ✅ `sentencepiece`: 0.1.99 → 0.2.0+ (fixes build errors on macOS)
- ✅ `fastapi`: 0.109.0 → 0.115.0+
- ✅ `pydantic`: 2.5.3 → 2.10.0+
- ✅ All other dependencies updated to latest compatible versions

## How to Apply

**If the service is currently running, you must restart it:**

```bash
# Stop the running service (Ctrl+C in the terminal where it's running)
# Then restart:
./run.sh
```

**For fresh setup:**
```bash
./setup.sh
./run.sh
```

## Verification

Test that everything works:

```bash
# Test the service
./test_translation.sh
```

Or manually:

```bash
# Check health
curl http://localhost:8000/api/v1/health

# Test translation
curl -X POST "http://localhost:8000/api/v1/translate" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, world!",
    "source_lang": "eng_Latn",
    "target_lang": "spa_Latn"
  }'
```

## Backwards Compatibility

The fix maintains backwards compatibility with older transformers versions by checking if the `lang_code_to_id` attribute exists before using it, falling back to the new API if not available.

## Next Steps

1. **Restart the service** if it's currently running
2. **Run tests** to verify everything works: `./test_translation.sh`
3. **Monitor logs** for any other issues

## Additional Scripts Created

- **[setup.sh](setup.sh)**: Automated setup script with proper dependency installation order
- **[run.sh](run.sh)**: Simple service startup script
- **[test_translation.sh](test_translation.sh)**: Automated testing script

---

**Fixed by:** Claude Code
**Date:** 2026-02-16
**Status:** ✅ Ready to deploy
