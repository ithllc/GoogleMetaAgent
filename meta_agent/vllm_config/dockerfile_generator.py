def generate_dockerfile(agent_name: str, requirements: list):
    """
    Generates a Dockerfile for the agent.
    """
    dockerfile_template = """\
# Start from a base image with CUDA and Python pre-installed.
FROM nvidia/cuda:12.1.0-base-ubuntu22.04

# Set up the environment.
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y python3-pip

# Copy the generated agent code and VLLM configuration scripts.
COPY . /app
WORKDIR /app

# Install all dependencies from a generated requirements.txt file.
RUN pip install --no-cache-dir -r requirements.txt

# Define the CMD to launch the VLLM server and the ADK FastAPI wrapper.
CMD ["python3", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
"""
    return dockerfile_template
