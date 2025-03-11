FILES = ./soika/src/*

lint:
	@echo "Running pylint check..."
	python -m pylint ${FILES}

format:
	@echo "Starting black formatting..."
	python -m black ${FILES}

test:
	@echo "Starting tests..."
	python -m pytest

activate:
	@echo "Activating virtual environment..."
	poetry shell

install: 
	@echo "Installing poetry dependencies..."
	poetry install
	poetry run pre-commit install

setup: install activate
finish: lint test 