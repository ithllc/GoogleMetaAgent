#!/bin/bash

# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Google Meta Agent Deployment Script
# Based on the QUICKSTART_GUIDE.md patterns

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ID=${GOOGLE_CLOUD_PROJECT:-""}
REGION=${GOOGLE_CLOUD_REGION:-"us-central1"}
SERVICE_NAME="google-meta-agent"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

echo -e "${BLUE}🚀 Google Meta Agent Deployment${NC}"
echo "========================================"

# Check prerequisites
check_prerequisites() {
    echo -e "${BLUE}⚡ Checking Prerequisites${NC}"
    
    if [ -z "$PROJECT_ID" ]; then
        echo -e "${RED}❌ PROJECT_ID not set. Please set GOOGLE_CLOUD_PROJECT environment variable${NC}"
        exit 1
    fi
    
    # Check if gcloud is installed
    if ! command -v gcloud &> /dev/null; then
        echo -e "${RED}❌ gcloud CLI not found. Please install Google Cloud SDK${NC}"
        exit 1
    fi
    
    # Check if docker is installed
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}❌ Docker not found. Please install Docker${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✅ Prerequisites checked${NC}"
}

# Setup Google Cloud environment
setup_gcloud() {
    echo -e "${BLUE}🔧 Setting up Google Cloud${NC}"
    
    # Set project and region
    gcloud config set project $PROJECT_ID
    gcloud config set run/region $REGION
    
    # Enable required APIs
    echo "Enabling required Google Cloud APIs..."
    gcloud services enable run.googleapis.com cloudbuild.googleapis.com aiplatform.googleapis.com
    
    echo -e "${GREEN}✅ Google Cloud setup complete${NC}"
}

# Build and push container
build_and_push() {
    echo -e "${BLUE}🏗️  Building and pushing container${NC}"
    
    # Build the container image
    echo "Building container image..."
    docker build -t $IMAGE_NAME .
    
    # Configure Docker for GCR
    gcloud auth configure-docker
    
    # Push the image
    echo "Pushing container image to Google Container Registry..."
    docker push $IMAGE_NAME
    
    echo -e "${GREEN}✅ Container built and pushed${NC}"
}

# Deploy to Cloud Run
deploy_to_cloud_run() {
    echo -e "${BLUE}🚀 Deploying to Cloud Run${NC}"
    
    # Deploy with GPU support
    gcloud run deploy $SERVICE_NAME \
        --image $IMAGE_NAME \
        --region $REGION \
        --platform managed \
        --allow-unauthenticated \
        --memory 32Gi \
        --cpu 8 \
        --gpu 1 \
        --gpu-type nvidia-l4 \
        --concurrency 1 \
        --max-instances 1 \
        --timeout 3600 \
        --no-cpu-throttling \
        --execution-environment gen2 \
        --port 8080 \
        --set-env-vars GOOGLE_CLOUD_PROJECT=$PROJECT_ID,GOOGLE_CLOUD_LOCATION=$REGION,CUDA_VISIBLE_DEVICES=0,NVIDIA_VISIBLE_DEVICES=0 \
        --labels dev-tutorial=hackathon-nyc-cloud-run-gpu-25
    
    echo -e "${GREEN}✅ Deployment complete${NC}"
}

# Get service URL
get_service_url() {
    echo -e "${BLUE}🌐 Getting service URL${NC}"
    
    SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --region=$REGION --format='value(status.url)')
    
    echo -e "${GREEN}🎉 Meta Agent deployed successfully!${NC}"
    echo -e "${GREEN}🔗 Service URL: $SERVICE_URL${NC}"
    echo -e "${GREEN}📊 Health check: $SERVICE_URL/health${NC}"
    
    export META_AGENT_URL=$SERVICE_URL
}

# Test the deployment
test_deployment() {
    echo -e "${BLUE}🧪 Testing deployment${NC}"
    
    if [ -z "$META_AGENT_URL" ]; then
        echo -e "${YELLOW}⚠️  META_AGENT_URL not set, skipping test${NC}"
        return
    fi
    
    echo "Testing health endpoint..."
    if curl -f "$META_AGENT_URL/health" > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Health check passed${NC}"
    else
        echo -e "${YELLOW}⚠️  Health check failed, but service may still be starting${NC}"
    fi
}

# Main deployment flow
main() {
    echo "Starting deployment with the following configuration:"
    echo "  Project ID: $PROJECT_ID"
    echo "  Region: $REGION"
    echo "  Service Name: $SERVICE_NAME"
    echo "  Image Name: $IMAGE_NAME"
    echo ""
    
    check_prerequisites
    setup_gcloud
    build_and_push
    deploy_to_cloud_run
    get_service_url
    test_deployment
    
    echo ""
    echo -e "${GREEN}🎊 Deployment completed successfully!${NC}"
    echo "Your Google Meta Agent is now running on Cloud Run with GPU support."
    echo ""
    echo "Next steps:"
    echo "1. Test your agent at: $SERVICE_URL"
    echo "2. Check logs: gcloud run services logs tail $SERVICE_NAME --region=$REGION"
    echo "3. Monitor performance in the Cloud Console"
    echo ""
    echo "To generate and deploy an agent, send a POST request to:"
    echo "  $SERVICE_URL/create-agent"
    echo ""
    echo -e "${BLUE}Happy hacking! 🚀${NC}"
}

# Handle script arguments
case "${1:-}" in
    "check")
        check_prerequisites
        ;;
    "setup")
        setup_gcloud
        ;;
    "build")
        build_and_push
        ;;
    "deploy")
        deploy_to_cloud_run
        get_service_url
        ;;
    "test")
        test_deployment
        ;;
    *)
        main
        ;;
esac
