.PHONY: help install install-dev test lint format demo serve clean docker-build docker-up

help:
	@echo "install      - install runtime package (editable)"
	@echo "install-dev  - install with dev deps"
	@echo "test         - run tests with coverage"
	@echo "lint         - run ruff lint"
	@echo "format       - run ruff format"
	@echo "demo         - run the end-to-end what-if demo"
	@echo "serve        - start the FastAPI service"
	@echo "clean        - remove caches and artifacts"
	@echo "docker-build - build the docker image"
	@echo "docker-up    - run via docker compose"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check src tests

format:
	ruff format src tests

demo:
	python scripts/run_demo.py

serve:
	uvicorn supply_twin.api.main:app --host 0.0.0.0 --port 8000

clean:
	python -c "import shutil,glob,os; [shutil.rmtree(p, ignore_errors=True) for p in glob.glob('**/__pycache__', recursive=True)+['.pytest_cache','.ruff_cache','htmlcov','.coverage','coverage.xml']]"

docker-build:
	docker build -t supply-twin:latest .

docker-up:
	docker compose up --build
