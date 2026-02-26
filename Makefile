.PHONY: help install install-dev run test lint format clean docker-build docker-up docker-down train

help:
	@echo "Available commands:"
	@echo "  make install      - Install production dependencies"
	@echo "  make install-dev  - Install development dependencies"
	@echo "  make run          - Run the application locally"
	@echo "  make test         - Run tests"
	@echo "  make lint         - Run linters"
	@echo "  make format       - Format code"
	@echo "  make clean        - Clean up cache and build files"
	@echo "  make docker-build - Build Docker image"
	@echo "  make docker-up    - Start Docker services"
	@echo "  make docker-down  - Stop Docker services"
	@echo "  make train        - Example training command"

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt

run:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

run-prod:
	gunicorn app.main:app \
		--workers 4 \
		--worker-class uvicorn.workers.UvicornWorker \
		--bind 0.0.0.0:8000 \
		--timeout 120

test:
	pytest -v --cov=app --cov-report=term-missing

test-verbose:
	pytest -vv --cov=app --cov-report=html

lint:
	flake8 app/ tests/
	mypy app/

format:
	black app/ tests/ training/
	isort app/ tests/ training/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.coverage" -delete
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf dist
	rm -rf build
	rm -rf *.egg-info

docker-build:
	docker-compose build

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

train:
	@echo "Example training command:"
	@echo "python training/train.py \\"
	@echo "  --data-file data/example_medical_en_hi.tsv \\"
	@echo "  --source-lang eng_Latn \\"
	@echo "  --target-lang hin_Deva \\"
	@echo "  --output-dir ./models/custom-nllb-medical \\"
	@echo "  --epochs 3 \\"
	@echo "  --batch-size 8"

evaluate:
	@echo "Example evaluation command:"
	@echo "python training/evaluate.py \\"
	@echo "  --model-path ./models/custom-nllb-medical \\"
	@echo "  --test-file data/example_medical_en_hi.tsv \\"
	@echo "  --source-lang eng_Latn \\"
	@echo "  --target-lang hin_Deva"

export-onnx:
	@echo "Example ONNX export command:"
	@echo "python training/export_onnx.py \\"
	@echo "  --model-path ./models/custom-nllb-medical \\"
	@echo "  --output-path ./models/onnx-medical \\"
	@echo "  --validate"
