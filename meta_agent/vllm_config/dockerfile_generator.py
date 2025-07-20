def generate_dockerfile(agent_name: str, requirements: list, use_gpu: bool = True, base_image: str = None):
    """
    Generates a Dockerfile for the agent.
    
    Args:
        agent_name: Name of the agent
        requirements: List of Python package requirements
        use_gpu: Whether to use GPU-enabled base image
        base_image: Custom base image (if None, uses default)
    """
    
    # Choose appropriate base image
    if base_image is None:
        if use_gpu:
            base_image = "nvidia/cuda:12.1.0-devel-ubuntu22.04"
        else:
            base_image = "python:3.11-slim"
    
    dockerfile_template = f"""\
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

# Base image with CUDA support for GPU inference
FROM {base_image}

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    python3 \\
    python3-pip \\
    python3-dev \\
    build-essential \\
    curl \\
    git \\
    && rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /app

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir --upgrade pip
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the agent code
COPY . .

# Create a non-root user for security
RUN useradd -m -u 1000 agent && chown -R agent:agent /app
USER agent

# Expose the port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \\
    CMD curl -f http://localhost:8080/health || exit 1

# Run the agent server
CMD ["python3", "server.py"]
"""
    
    return dockerfile_template

def generate_requirements_txt(base_requirements: list = None, include_vllm: bool = True, include_adk: bool = True):
    """
    Generate requirements.txt content for the agent.
    
    Args:
        base_requirements: List of additional requirements
        include_vllm: Whether to include vLLM dependencies
        include_adk: Whether to include Google ADK dependencies
    """
    requirements = []
    
    # Core dependencies
    requirements.extend([
        "fastapi>=0.104.0",
        "uvicorn[standard]>=0.24.0",
        "pydantic>=2.0.0",
        "requests>=2.31.0",
        "python-dotenv>=1.0.0",
    ])
    
    # Google Cloud and ADK dependencies
    if include_adk:
        requirements.extend([
            "google-adk>=1.0.0",
            "google-cloud-run>=0.10.0",
            "google-cloud-build>=3.20.0",
            "google-auth>=2.23.0",
            "google-auth-oauthlib>=1.1.0",
        ])
    
    # vLLM dependencies
    if include_vllm:
        requirements.extend([
            "vllm>=0.2.7",
            "torch>=2.1.0",
            "transformers>=4.36.0",
            "tokenizers>=0.15.0",
        ])
    
    # Monitoring
    requirements.extend([
        "opik>=0.1.0",
    ])
    
    # Add custom requirements
    if base_requirements:
        requirements.extend(base_requirements)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_requirements = []
    for req in requirements:
        if req not in seen:
            seen.add(req)
            unique_requirements.append(req)
    
    return "\\n".join(unique_requirements)

def generate_cloudbuild_yaml(project_id: str, agent_name: str, region: str = "us-central1"):
    """
    Generate Cloud Build configuration YAML.
    """
    cloudbuild_yaml = f"""\
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

steps:
  # Build the container image
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', 'gcr.io/{project_id}/{agent_name}:$COMMIT_SHA', '.']
    
  # Push the container image to Container Registry
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', 'gcr.io/{project_id}/{agent_name}:$COMMIT_SHA']
    
  # Deploy to Cloud Run
  - name: 'gcr.io/cloud-builders/gcloud'
    args:
      - 'run'
      - 'deploy'
      - '{agent_name}'
      - '--image=gcr.io/{project_id}/{agent_name}:$COMMIT_SHA'
      - '--region={region}'
      - '--platform=managed'
      - '--allow-unauthenticated'
      - '--memory=4Gi'
      - '--cpu=2'
      - '--concurrency=1'
      - '--max-instances=1'
      - '--gpu=1'
      - '--gpu-type=nvidia-l4'

images:
  - 'gcr.io/{project_id}/{agent_name}:$COMMIT_SHA'

options:
  logging: CLOUD_LOGGING_ONLY
  machineType: 'E2_HIGHCPU_8'
"""
    
    return cloudbuild_yaml
