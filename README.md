# 🤖 Google Meta Agent

**Automated Design of Agentic Systems (ADAS) for Google Cloud Run with GPUs**

The Google Meta Agent is an innovative implementation of the Automated Design of Agentic Systems (ADAS) concept, where a meta agent automatically generates, builds, and deploys other AI agents based on natural language descriptions. This project demonstrates the future of AI development by moving from hand-crafting agents to defining high-level goals and letting autonomous systems handle the implementation.

## 🌟 Overview

This project implements the concepts from the "Automated Design of Agentic Systems" research, leveraging:

- **Google ADK (Agent Development Kit)** for robust agent framework
- **DSPy** for structured language model programming
- **Google Cloud Run with GPU support** for scalable deployment
- **VLLM** for efficient model inference
- **Opik by Comet** for monitoring and observability

## 🏗️ Architecture

The Meta Agent follows a sophisticated pipeline:

```
Natural Language Description → Agent Blueprint → Code Generation → Containerization → Cloud Deployment
```

### Key Components

1. **Agent Generation Module** (`meta_agent/agent_generation/`)
   - Uses DSPy and Pydantic to create structured agent blueprints
   - Generates agent code with proper ADK integration
   - Supports multiple tool types (Google Search, Code Executor, etc.)

2. **VLLM Configuration Module** (`meta_agent/vllm_config/`)
   - Manages VLLM instances for local model inference
   - GPU detection and optimization
   - Load balancing for multiple model instances

3. **Deployment Orchestration** (`meta_agent/deployment/`)
   - Automated Cloud Build integration
   - Cloud Run deployment with GPU support
   - Service configuration and scaling

## 🚀 Quick Start

### Prerequisites

- Google Cloud project with billing enabled
- Docker installed
- Google Cloud SDK (gcloud) installed
- Python 3.11+ with pip

### 1. Setup Environment Using Google Cloud Shell

```bash
# Clone the repository
git clone https://github.com/ithllc/GoogleMetaAgent.git
cd GoogleMetaAgent

# Set up Google Cloud
export GOOGLE_CLOUD_PROJECT="your-project-id"
gcloud config set project $GOOGLE_CLOUD_PROJECT
gcloud config set run/region us-central1

# Enable required APIs
gcloud services enable run.googleapis.com cloudbuild.googleapis.com aiplatform.googleapis.com
```

### 2. Install Dependencies

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 3. Deploy Meta Agent

```bash
# Make deployment script executable
chmod +x deploy.sh

# Run deployment
./deploy.sh
```

### 4. Use the Meta Agent

Once deployed, you can create new agents via the API:

```bash
# Example: Create a RAG agent
curl -X POST "https://your-meta-agent-url/create-agent" \
  -H "Content-Type: application/json" \
  -d '{
    "description": "A RAG agent that searches the web and answers questions about current events"
  }'
```

## 📊 API Endpoints

### Core Endpoints

- `POST /create-agent` - Generate a new agent from description
- `GET /list-agents` - List all generated agents  
- `GET /agent/{name}/files` - Get files for a specific agent
- `GET /health` - Health check endpoint

### Example Request

```json
{
  "description": "A math agent that can solve equations and perform calculations using code execution",
  "output_directory": "custom_agents/math_solver"
}
```

### Example Response

```json
{
  "success": true,
  "agent_name": "math_agent",
  "service_url": "https://math-agent-project.us-central1.run.app",
  "output_directory": "generated_agents/math_agent",
  "deployment_commands": {
    "build": "docker build -t gcr.io/project/math-agent generated_agents/math_agent",
    "push": "docker push gcr.io/project/math-agent",
    "deploy": "gcloud run deploy math-agent --image gcr.io/project/math-agent --region us-central1 --gpu 1"
  },
  "message": "Agent 'math_agent' generated successfully!"
}
```

## 🛠️ Generated Agent Structure

Each generated agent includes:

```
generated_agents/agent_name/
├── agent.py          # Main agent implementation
├── server.py         # FastAPI server wrapper
├── __init__.py       # Package initialization
├── requirements.txt  # Python dependencies
├── Dockerfile        # Container configuration
└── cloudbuild.yaml   # Cloud Build configuration
```

## 🔧 Configuration

### Environment Variables

- `GOOGLE_CLOUD_PROJECT` - Your GCP project ID
- `GOOGLE_CLOUD_LOCATION` - Deployment region (default: us-central1)
- `PORT` - Server port (default: 8080)
- `OPIK_API_KEY` - Opik monitoring API key (optional)

### Cloud Run Configuration

The Meta Agent is configured for GPU-enabled Cloud Run:

- **GPU**: 1x NVIDIA L4
- **Memory**: 32Gi
- **CPU**: 8 cores
- **Timeout**: 3600 seconds
- **Concurrency**: 1

## 📈 Monitoring

The system integrates with **Opik by Comet** for comprehensive monitoring:

- Agent execution traces
- Performance metrics
- Error tracking
- Usage analytics

## 🧪 Testing

Run the test suite:

```bash
# Unit tests
python -m pytest tests/test_agent_generation.py

# Integration tests
python -m pytest tests/test_integration.py
```

## 🎯 Examples

### Create a Research Assistant

```bash
curl -X POST "https://your-meta-agent-url/create-agent" \
  -H "Content-Type: application/json" \
  -d '{
    "description": "A research assistant that can search academic papers, summarize findings, and help with literature reviews"
  }'
```

### Create a Code Review Agent

```bash
curl -X POST "https://your-meta-agent-url/create-agent" \
  -H "Content-Type: application/json" \
  -d '{
    "description": "A code review agent that analyzes code, suggests improvements, and checks for security vulnerabilities"
  }'
```

## 🔐 Security

- All services run with least-privilege principles
- Container images use non-root users
- API endpoints include proper authentication
- Environment variables for sensitive configuration

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📚 References

- [Automated Design of Agentic Systems Paper](https://arxiv.org/abs/2408.08435v2)
- [Google ADK Documentation](https://google.github.io/adk-docs/)
- [Cloud Run GPU Documentation](https://cloud.google.com/run/docs/configuring/gpu)
- [VLLM Documentation](https://vllm.readthedocs.io/)
- [DSPy Documentation](https://dspy.ai/)

## 📄 License

This project is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)** - see the [LICENSE](LICENSE) file for details.

### AGPL-3.0 License Summary

The AGPL-3.0 is a strong copyleft license that ensures:

- **Freedom to use**: You can use this software for any purpose
- **Freedom to study**: You can examine and modify the source code
- **Freedom to distribute**: You can redistribute the software
- **Network use provision**: If you run this software on a server and provide services to users, you must make the source code available to those users

**Key Requirements:**
- Any modifications or derivative works must also be licensed under AGPL-3.0
- If you deploy this software as a network service, you must provide source code access to users
- You must preserve copyright notices and license information
- You must include the license text with any distribution

For more information about AGPL-3.0, visit: https://www.gnu.org/licenses/agpl-3.0.html

## 🏆 Hackathon Context

This project was developed for the **Agentic AI App Hackathon with Google Cloud Run GPUs**, demonstrating:

- ✅ End-to-end agent automation
- ✅ Cloud Run GPU utilization  
- ✅ Open model integration
- ✅ Scalable architecture
- ✅ Production-ready deployment

---

**Built with ❤️ for the Google Cloud Run GPU Hackathon 2025**
