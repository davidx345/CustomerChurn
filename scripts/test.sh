# scripts/test.sh
#!/bin/bash
# Comprehensive testing script
# [Intent] Run all tests with coverage reporting
# [DevOps] CI/CD integration and quality gates

set -e

echo "🧪 Running comprehensive test suite..."

# Setup test environment
echo "🔧 Setting up test environment..."
python -m venv test_env
source test_env/bin/activate || test_env/Scripts/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Code quality checks
echo "📏 Running code quality checks..."

# Linting
echo "🔍 Running flake8 linting..."
flake8 src/ tests/ --max-line-length=100 --exclude=__pycache__

# Security scanning
echo "🔒 Running security scan..."
bandit -r src/ -f json -o security-report.json || true
safety check --json --output safety-report.json || true

# Type checking
echo "🏷️ Running type checking..."
mypy src/ --ignore-missing-imports || true

# Unit tests with coverage
echo "🧪 Running unit tests with coverage..."
pytest tests/ -v --cov=src --cov-report=html --cov-report=json --cov-report=term

# Integration tests
echo "🔗 Running integration tests..."
python -m pytest tests/test_api.py -v

# Load testing (if application is running)
echo "⚡ Running load tests..."
if curl -s http://localhost:5000/health > /dev/null; then
  locust -f tests/load_test.py --headless -u 10 -r 2 -t 30s --host=http://localhost:5000
else
  echo "⚠️ Application not running, skipping load tests"
fi

# Generate test report
echo "📊 Generating test report..."
echo "Test Summary:" > test-report.txt
echo "=============" >> test-report.txt
echo "Timestamp: $(date)" >> test-report.txt
echo "Linting: $(flake8 src/ --count 2>/dev/null || echo 'Issues found')" >> test-report.txt
echo "Security: Check security-report.json" >> test-report.txt
echo "Coverage: Check htmlcov/index.html" >> test-report.txt

# Cleanup
deactivate
rm -rf test_env

echo "✅ All tests completed! Check test-report.txt for summary."
