# Enhanced Makefile for comprehensive project management
# [Intent] One-command operations for development and deployment
# [DevOps] Standardized workflows for all environments

.PHONY: help install test lint security build run deploy clean docs

# Default target
help:
	@echo "Available commands:"
	@echo "  install     - Install all dependencies"
	@echo "  test        - Run comprehensive test suite"
	@echo "  lint        - Run code quality checks"
	@echo "  security    - Run security scans"
	@echo "  build       - Build Docker image"
	@echo "  run         - Run application locally"
	@echo "  deploy      - Deploy to production"
	@echo "  monitor     - Start monitoring stack"
	@echo "  clean       - Clean up generated files"
	@echo "  docs        - Generate documentation"

install:
	@echo "📦 Installing dependencies..."
	pip install -r requirements.txt
	pip install -r requirements-dev.txt
	@echo "✅ Dependencies installed"

test:
	@echo "🧪 Running test suite..."
	python -m pytest tests/ -v --cov=src --cov-report=html --cov-report=term
	@echo "✅ Tests completed"

lint:
	@echo "🔍 Running code quality checks..."
	flake8 src/ tests/ --max-line-length=100
	black --check src/ tests/
	mypy src/ --ignore-missing-imports
	@echo "✅ Linting completed"

security:
	@echo "🔒 Running security scans..."
	bandit -r src/ -f json -o reports/security.json
	safety check --json --output reports/safety.json
	@echo "✅ Security scan completed"

format:
	@echo "🎨 Formatting code..."
	black src/ tests/
	@echo "✅ Code formatted"

build:
	@echo "🐳 Building Docker image..."
	docker build -t customer-churn:latest .
	@echo "✅ Docker image built"

run:
	@echo "🚀 Starting application..."
	python app_new.py

run-docker:
	@echo "🐳 Running Docker container..."
	docker run -p 5000:5000 -e FLASK_ENV=development customer-churn:latest

deploy:
	@echo "🚀 Deploying to production..."
	./scripts/deploy.sh

monitor:
	@echo "📊 Starting monitoring stack..."
	cd monitoring && docker-compose -f docker-compose.monitoring.yml up -d
	@echo "✅ Monitoring stack started"
	@echo "📊 Grafana: http://localhost:3000 (admin/admin123)"
	@echo "📊 Prometheus: http://localhost:9090"

stop-monitor:
	@echo "🛑 Stopping monitoring stack..."
	cd monitoring && docker-compose -f docker-compose.monitoring.yml down

clean:
	@echo "🧹 Cleaning up..."
	rm -rf __pycache__ *.pyc *.pyo *.pyd .pytest_cache .coverage htmlcov
	rm -rf reports/*.json
	docker system prune -f
	@echo "✅ Cleanup completed"

docs:
	@echo "📚 Generating documentation..."
	mkdir -p docs
	python -c "import src; help(src)" > docs/api.txt
	@echo "✅ Documentation generated"

setup-dev:
	@echo "🔧 Setting up development environment..."
	make install
	pre-commit install
	mkdir -p reports logs
	@echo "✅ Development environment ready"

ci-test:
	@echo "🤖 Running CI test pipeline..."
	make lint
	make security
	make test
	@echo "✅ CI pipeline completed"

load-test:
	@echo "⚡ Running load tests..."
	locust -f tests/load_test.py --headless -u 50 -r 10 -t 60s --host=http://localhost:5000

backup:
	@echo "💾 Creating backup..."
	tar -czf backup-$(shell date +%Y%m%d-%H%M%S).tar.gz src/ tests/ config/ static/ templates/ *.py *.md
	@echo "✅ Backup created"
