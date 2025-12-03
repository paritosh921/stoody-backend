#!/bin/bash

# SkillBot Async Backend Deployment Script
# For EC2 deployment with automatic setup

set -e

echo "🚀 SkillBot Async Backend Deployment Script"
echo "==========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   print_error "This script should not be run as root"
   exit 1
fi

# Check for required arguments
if [ "$#" -ne 1 ]; then
    print_error "Usage: $0 <production|staging>"
    exit 1
fi

ENVIRONMENT=$1

print_status "Deploying to: $ENVIRONMENT"

# Update system packages
print_status "Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install required system packages
print_status "Installing system dependencies..."
sudo apt install -y python3.9 python3.9-venv python3-pip nginx supervisor redis-server git curl htop

# Configure Redis
print_status "Configuring Redis..."
sudo systemctl enable redis-server
sudo systemctl start redis-server

# Create application directory
APP_DIR="/home/$USER/skillbot-backend"
print_status "Setting up application directory: $APP_DIR"

if [ ! -d "$APP_DIR" ]; then
    mkdir -p "$APP_DIR"
fi

cd "$APP_DIR"

# Clone or update repository
if [ -d ".git" ]; then
    print_status "Updating existing repository..."
    git pull origin main
else
    print_status "Cloning repository..."
    git clone https://github.com/yourusername/hustle-aid.git .
fi

cd backend

# Create virtual environment
print_status "Setting up Python virtual environment..."
if [ ! -d "venv" ]; then
    python3.9 -m venv venv
fi

source venv/bin/activate

# Install Python dependencies
print_status "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
pip install uvloop httptools gunicorn  # Performance optimizations

# Create environment file
print_status "Setting up environment configuration..."
if [ ! -f ".env" ]; then
    cat > .env << EOF
# Environment
NODE_ENV=$ENVIRONMENT
PORT=5001
HOST=127.0.0.1

# Database Configuration
MONGODB_URI=mongodb://localhost:27017/skillbot_$ENVIRONMENT
MONGODB_DB_NAME=skillbot_$ENVIRONMENT
MONGODB_MIN_POOL_SIZE=150
MONGODB_MAX_POOL_SIZE=1000

# Redis Configuration
REDIS_URL=redis://localhost:6379/0
CACHE_TTL=3600
CACHE_MAX_CONNECTIONS=500

# Authentication (CHANGE THESE IN PRODUCTION!)
JWT_SECRET_KEY=change-this-in-production-$(openssl rand -base64 32)
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=60

# OpenAI Configuration (SET YOUR API KEY!)
OPENAI_API_KEY=your-openai-api-key-here
OPENAI_MODEL=gpt-4-vision-preview
OPENAI_MAX_TOKENS=1000
OPENAI_TEMPERATURE=0.7
OPENAI_CONCURRENCY_LIMIT=200
OCR_CONCURRENCY_LIMIT=8

# Rate Limiting
RATE_LIMIT_DEFAULT=600/minute
RATE_LIMIT_AUTH=120/minute
RATE_LIMIT_UPLOAD=120/minute
API_RATE_LIMIT_RPM=600
API_RATE_LIMIT_BURST=200

# Performance
MAX_WORKERS=8
WORKER_CONNECTIONS=2000
EOF

    print_warning "Environment file created. Please update .env with your actual configuration!"
else
    print_status "Environment file already exists"
fi

# Load environment variables for deployment defaults
if [ -f "$APP_DIR/backend/.env" ]; then
    set -a
    # shellcheck disable=SC1090
    source "$APP_DIR/backend/.env"
    set +a
fi

API_RATE_LIMIT_RPM=${API_RATE_LIMIT_RPM:-600}
API_RATE_LIMIT_BURST=${API_RATE_LIMIT_BURST:-200}
NGINX_API_RATE="${API_RATE_LIMIT_RPM}r/m"
NGINX_API_BURST=${API_RATE_LIMIT_BURST}
MAX_WORKERS=${MAX_WORKERS:-8}
WORKER_CONNECTIONS=${WORKER_CONNECTIONS:-2000}

# Create necessary directories
print_status "Creating application directories..."
mkdir -p chromadb_data images logs

# Set up Supervisor configuration
print_status "Configuring Supervisor..."
sudo tee /etc/supervisor/conf.d/skillbot-$ENVIRONMENT.conf > /dev/null << EOF
[program:skillbot-$ENVIRONMENT]
command=$APP_DIR/backend/venv/bin/python -m uvicorn main_async:app --host 0.0.0.0 --port 5001 --workers $MAX_WORKERS --limit-concurrency $WORKER_CONNECTIONS
directory=$APP_DIR/backend
user=$USER
autostart=true
autorestart=true
redirect_stderr=true
stdout_logfile=/var/log/supervisor/skillbot-$ENVIRONMENT.log
stdout_logfile_maxbytes=50MB
stdout_logfile_backups=10
environment=PATH="$APP_DIR/backend/venv/bin"
EOF

# Set up Nginx configuration
print_status "Configuring Nginx..."
sudo tee /etc/nginx/sites-available/skillbot-$ENVIRONMENT > /dev/null << EOF
upstream skillbot_$ENVIRONMENT {
    least_conn;
    server 127.0.0.1:5001 max_fails=3 fail_timeout=30s;
}

server {
    listen 80;
    server_name localhost;

    # Rate limiting
    limit_req_zone \$binary_remote_addr zone=api_$ENVIRONMENT:10m rate=$NGINX_API_RATE;

    # API endpoints
    location /api/ {
        limit_req zone=api_$ENVIRONMENT burst=$NGINX_API_BURST nodelay;

        proxy_pass http://skillbot_$ENVIRONMENT;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_cache_bypass \$http_upgrade;

        # Timeouts
        proxy_connect_timeout 5s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }

    # Health check
    location /health {
        proxy_pass http://skillbot_$ENVIRONMENT;
        access_log off;
    }

    # Root endpoint
    location / {
        proxy_pass http://skillbot_$ENVIRONMENT;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }

    # Static files
    location /images/ {
        alias $APP_DIR/backend/images/;
        expires 1d;
        add_header Cache-Control "public, immutable";
    }
}
EOF

# Enable Nginx site
sudo ln -sf /etc/nginx/sites-available/skillbot-$ENVIRONMENT /etc/nginx/sites-enabled/
sudo nginx -t

# Update Supervisor and Nginx
print_status "Starting services..."
sudo supervisorctl reread
sudo supervisorctl update
sudo systemctl reload nginx

# Start the application
sudo supervisorctl start skillbot-$ENVIRONMENT

# Create monitoring script
print_status "Setting up monitoring..."
cat > monitor.py << 'EOF'
#!/usr/bin/env python3
import requests
import time
import sys
import json

def check_health():
    try:
        response = requests.get('http://localhost/health', timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Backend healthy - Status: {data.get('status', 'unknown')}")
            return True
        else:
            print(f"❌ Backend unhealthy - HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Backend unreachable - {str(e)}")
        return False

def check_api():
    try:
        response = requests.post('http://localhost/api/v1/chat',
            json={
                "message": "Health check test",
                "sessionId": "health_check",
                "userId": "monitor"
            },
            timeout=15
        )
        if response.status_code == 200:
            print("✅ API responding normally")
            return True
        else:
            print(f"⚠️  API returned HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ API test failed - {str(e)}")
        return False

if __name__ == "__main__":
    print(f"🔍 Health check at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    health_ok = check_health()
    api_ok = check_api()

    if health_ok and api_ok:
        print("✅ All systems operational")
        sys.exit(0)
    else:
        print("❌ System issues detected")
        sys.exit(1)
EOF

chmod +x monitor.py

# Set up system optimization
print_status "Applying system optimizations..."
sudo tee -a /etc/security/limits.conf > /dev/null << EOF
$USER soft nofile 65536
$USER hard nofile 65536
EOF

# Check final status
print_status "Checking deployment status..."
sleep 5

if curl -f http://localhost/health > /dev/null 2>&1; then
    print_status "🎉 Deployment successful!"
    echo ""
    echo "📊 Status Check:"
    python3 monitor.py
    echo ""
    echo "🔗 API Endpoints:"
    echo "   Health: http://localhost/health"
    echo "   Chat: http://localhost/api/v1/chat"
    echo "   Docs: http://localhost/docs (if debug mode)"
    echo ""
    echo "📝 Next Steps:"
    echo "   1. Update .env with your actual API keys"
    echo "   2. Configure your domain name in Nginx"
    echo "   3. Set up SSL with Let's Encrypt"
    echo "   4. Run load tests: python test_concurrency.py"
    echo ""
    echo "📋 Management Commands:"
    echo "   Start: sudo supervisorctl start skillbot-$ENVIRONMENT"
    echo "   Stop: sudo supervisorctl stop skillbot-$ENVIRONMENT"
    echo "   Restart: sudo supervisorctl restart skillbot-$ENVIRONMENT"
    echo "   Logs: sudo tail -f /var/log/supervisor/skillbot-$ENVIRONMENT.log"

else
    print_error "Deployment failed - backend is not responding"
    echo "📋 Troubleshooting:"
    echo "   Check logs: sudo tail -f /var/log/supervisor/skillbot-$ENVIRONMENT.log"
    echo "   Check status: sudo supervisorctl status"
    echo "   Check nginx: sudo nginx -t"
    exit 1
fi
