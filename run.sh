#!/bin/bash

# NLLB Translation Service Run Script

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Starting NLLB Translation Service...${NC}"

# Activate virtual environment
source venv/bin/activate

# Check if .env exists
if [ ! -f .env ]; then
    echo "⚠️  .env file not found. Creating from .env.example..."
    cp .env.example .env
fi

# Run the application
echo -e "${GREEN}✅ Starting server on http://localhost:8000${NC}"
echo -e "${GREEN}📚 API Documentation: http://localhost:8000/docs${NC}"
echo ""

python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
