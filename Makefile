# Convenience targets — everything runs OFFLINE by default.
.PHONY: install install-dev data test lint eval serve docker-build docker-run clean

install:
	pip install -r requirements.txt

install-dev: install
	pip install -r requirements-dev.txt

data:
	python data/generate_data.py --set both

test:
	pytest --cov --cov-fail-under=85

lint:
	ruff check .
	ruff format --check .

eval:
	python run_all.py

serve:
	uvicorn app:app --host 0.0.0.0 --port 8000

docker-build:
	docker build -t affordability-copilot .

docker-run:
	docker run --rm -p 8000:8000 affordability-copilot

clean:
	rm -rf __pycache__ src/__pycache__ tests/__pycache__ .pytest_cache .coverage index
