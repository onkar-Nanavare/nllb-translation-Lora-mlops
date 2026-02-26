#!/bin/bash
# Test script for API endpoints

set -e

BASE_URL="${BASE_URL:-http://localhost:8000}"
API_PREFIX="/api/v1"

echo "=========================================="
echo "Testing NLLB Translation Service API"
echo "Base URL: $BASE_URL"
echo "=========================================="
echo

# Test 1: Health Check
echo "1. Testing Health Check..."
curl -s "$BASE_URL$API_PREFIX/health" | jq '.'
echo
echo

# Test 2: Get Languages
echo "2. Testing Get Languages (first 5)..."
curl -s "$BASE_URL$API_PREFIX/languages" | jq '.languages[:5]'
echo
echo

# Test 3: Simple Translation
echo "3. Testing Simple Translation (English to Hindi)..."
curl -s -X POST "$BASE_URL$API_PREFIX/translate" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, how are you?",
    "source_lang": "eng_Latn",
    "target_lang": "hin_Deva"
  }' | jq '.'
echo
echo

# Test 4: Translation with Glossary
echo "4. Testing Translation with Glossary..."
curl -s -X POST "$BASE_URL$API_PREFIX/translate" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The patient has hypertension",
    "source_lang": "eng_Latn",
    "target_lang": "hin_Deva",
    "glossary": {
      "hypertension": "उच्च रक्तचाप",
      "patient": "रोगी"
    }
  }' | jq '.'
echo
echo

# Test 5: Batch Translation
echo "5. Testing Batch Translation..."
curl -s -X POST "$BASE_URL$API_PREFIX/batch-translate" \
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
      },
      {
        "text": "Thank you",
        "source_lang": "eng_Latn",
        "target_lang": "fra_Latn"
      }
    ]
  }' | jq '.'
echo
echo

echo "=========================================="
echo "All tests completed!"
echo "=========================================="
