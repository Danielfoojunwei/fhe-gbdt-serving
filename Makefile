.PHONY: build test docker-up cookbook e2e-real bench-cookbook clean

build:
	@echo "Building services..."
	docker-compose build

docker-up:
	@echo "Starting services..."
	docker-compose up -d

stop:
	@echo "Stopping services..."
	docker-compose down

test:
	@echo "Running unit tests..."
	python -m pytest tests/unit

cookbook:
	@echo "Running all cookbook recipes..."
	python bench/cookbook/run_all.py

cookbook-xgboost:
	python bench/cookbook/run_recipe_xgboost.py

cookbook-lightgbm:
	python bench/cookbook/run_recipe_lightgbm.py

cookbook-catboost:
	python bench/cookbook/run_recipe_catboost.py

e2e-real:
	@echo "Running E2E tests with real models..."
	python -m pytest tests/e2e/real_models/

bench-cookbook: cookbook
	@echo "Benchmark reports generated in bench/reports/cookbook/"

clean:
	rm -rf bench/reports/*
	find . -name "__pycache__" -type d -exec rm -rf {} +
