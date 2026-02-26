#!/bin/bash

# Test script for NLLB Translation Service

echo "🧪 Testing NLLB Translation Service..."
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if service is running
echo -e "${BLUE}1. Checking health endpoint...${NC}"
HEALTH_RESPONSE=$(curl -s http://localhost:8000/api/v1/health)
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Service is running${NC}"
    echo "$HEALTH_RESPONSE" | python3 -m json.tool
else
    echo -e "${RED}✗ Service is not running. Start it with ./run.sh${NC}"
    exit 1
fi

echo ""
echo -e "${BLUE}2. Testing translation (English to Spanish)...${NC}"
TRANSLATE_RESPONSE=$(curl -s -X POST "http://localhost:8000/api/v1/translate" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, how are you today?",
    "source_lang": "eng_Latn",
    "target_lang": "spa_Latn"
  }')

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Translation successful${NC}"
    echo "$TRANSLATE_RESPONSE" | python3 -m json.tool
else
    echo -e "${RED}✗ Translation failed${NC}"
    exit 1
fi

echo ""
echo -e "${BLUE}3. Testing batch translation...${NC}"
BATCH_RESPONSE=$(curl -s -X POST "http://localhost:8000/api/v1/translate/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["Hello", "Goodbye", "Thank you"],
    "source_lang": "eng_Latn",
    "target_lang": "fra_Latn"
  }')

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Batch translation successful${NC}"
    echo "$BATCH_RESPONSE" | python3 -m json.tool
else
    echo -e "${RED}✗ Batch translation failed${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}✅ All tests passed!${NC}"
