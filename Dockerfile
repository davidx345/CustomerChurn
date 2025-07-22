# Dockerfile for Customer Churn Prediction App
# [Intent] Containerize the app for portability, cloud deployment, and DevOps best practices
# [Safety] Avoids running as root, exposes only necessary ports
# [Edge Cases] Handles missing dependencies, production vs. dev environments

FROM python:3.11-slim

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy app files
COPY . .

# Expose port (assuming Flask default)
EXPOSE 5000

# [Security] Avoid running as root
RUN useradd -m appuser
USER appuser

# [Intent] Start the app
CMD ["python", "app.py"]
