# scripts/deploy.sh
#!/bin/bash
# Production deployment script
# [Intent] Automated deployment to cloud platforms
# [DevOps] One-click deployment with health checks

set -e

# Configuration
APP_NAME="customer-churn"
DOCKER_IMAGE="$APP_NAME:latest"
HEALTH_CHECK_URL="http://localhost:5000/health"
MAX_WAIT_TIME=120

echo "🚀 Starting deployment of $APP_NAME..."

# Build Docker image
echo "📦 Building Docker image..."
docker build -t $DOCKER_IMAGE .

# Stop existing container
echo "🛑 Stopping existing container..."
docker stop $APP_NAME || true
docker rm $APP_NAME || true

# Start new container
echo "🏃 Starting new container..."
docker run -d \
  --name $APP_NAME \
  -p 5000:5000 \
  -e FLASK_ENV=production \
  -e LOG_LEVEL=INFO \
  --restart unless-stopped \
  $DOCKER_IMAGE

# Health check
echo "🏥 Performing health check..."
for i in $(seq 1 $MAX_WAIT_TIME); do
  if curl -s $HEALTH_CHECK_URL > /dev/null; then
    echo "✅ Application is healthy!"
    break
  fi
  
  if [ $i -eq $MAX_WAIT_TIME ]; then
    echo "❌ Health check failed after $MAX_WAIT_TIME seconds"
    docker logs $APP_NAME
    exit 1
  fi
  
  echo "⏳ Waiting for application to start... ($i/$MAX_WAIT_TIME)"
  sleep 1
done

echo "🎉 Deployment completed successfully!"
echo "📊 Application metrics: http://localhost:5000/metrics"
echo "🏥 Health check: $HEALTH_CHECK_URL"
