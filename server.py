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

import os
import logging
import traceback
from pathlib import Path
from typing import Dict, Any, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from meta_agent.main import MetaAgent

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Google Meta Agent",
    description="Automated Design of Agentic Systems (ADAS) - A meta agent that generates, builds, and deploys other AI agents",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Initialize Meta Agent
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT")
REGION = os.environ.get("GOOGLE_CLOUD_LOCATION", "europe-west1")

if not PROJECT_ID:
    logger.error("GOOGLE_CLOUD_PROJECT environment variable not set")
    raise ValueError("GOOGLE_CLOUD_PROJECT must be set")

meta_agent = MetaAgent(project_id=PROJECT_ID, location=REGION)

# Request/Response Models
class AgentCreationRequest(BaseModel):
    """Request model for agent creation."""
    description: str = Field(
        ..., 
        description="Natural language description of the desired agent",
        min_length=10,
        max_length=1000,
        example="A RAG agent that can search the web and answer questions about current events"
    )
    output_directory: Optional[str] = Field(
        None,
        description="Optional custom directory for generated agent files"
    )

class AgentCreationResponse(BaseModel):
    """Response model for agent creation."""
    success: bool = Field(..., description="Whether the agent creation was successful")
    agent_name: str = Field(..., description="Name of the generated agent")
    service_url: str = Field(..., description="Predicted Cloud Run service URL")
    output_directory: str = Field(..., description="Directory containing generated files")
    deployment_commands: Dict[str, str] = Field(..., description="Commands to deploy the agent")
    message: str = Field(..., description="Status message")

class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str = Field(..., description="Health status")
    service: str = Field(..., description="Service name")
    project_id: str = Field(..., description="Google Cloud Project ID")
    region: str = Field(..., description="Deployment region")

class AgentListResponse(BaseModel):
    """Response model for listing generated agents."""
    agents: list[str] = Field(..., description="List of generated agent names")
    count: int = Field(..., description="Number of agents")

# API Endpoints

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with service information."""
    return {
        "service": "Google Meta Agent",
        "description": "Automated Design of Agentic Systems (ADAS)",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        service="google-meta-agent",
        project_id=PROJECT_ID,
        region=REGION
    )

@app.post("/create-agent", response_model=AgentCreationResponse, tags=["Agent Management"])
async def create_agent(
    request: AgentCreationRequest,
    background_tasks: BackgroundTasks
) -> AgentCreationResponse:
    """
    Create a new AI agent based on a natural language description.
    
    This endpoint implements the core ADAS (Automated Design of Agentic Systems) functionality
    where the meta agent automatically generates, builds, and prepares an agent for deployment.
    """
    try:
        logger.info(f"Received agent creation request: {request.description}")
        
        # Generate the agent using the Meta Agent
        service_url = meta_agent.create_and_deploy_agent(
            description=request.description,
            agent_output_dir=request.output_directory
        )
        
        # Extract agent name from the service URL or generate from description
        agent_name = request.description.lower().replace(" ", "_").replace("-", "_")[:20]
        agent_name = "".join(c for c in agent_name if c.isalnum() or c == "_")
        
        # Determine output directory
        output_dir = request.output_directory or f"generated_agents/{agent_name}"
        
        # Generate deployment commands
        deployment_commands = {
            "build": f"docker build -t gcr.io/{PROJECT_ID}/{agent_name} {output_dir}",
            "push": f"docker push gcr.io/{PROJECT_ID}/{agent_name}",
            "deploy": f"gcloud run deploy {agent_name} --image gcr.io/{PROJECT_ID}/{agent_name} --region {REGION} --allow-unauthenticated --gpu 1 --gpu-type nvidia-l4"
        }
        
        return AgentCreationResponse(
            success=True,
            agent_name=agent_name,
            service_url=service_url,
            output_directory=output_dir,
            deployment_commands=deployment_commands,
            message=f"Agent '{agent_name}' generated successfully! Check the output directory for files."
        )
        
    except Exception as e:
        logger.error(f"Error creating agent: {str(e)}")
        logger.error(traceback.format_exc())
        
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create agent: {str(e)}"
        )

@app.get("/list-agents", response_model=AgentListResponse, tags=["Agent Management"])
async def list_agents():
    """List all generated agents."""
    try:
        agents_dir = Path("generated_agents")
        if not agents_dir.exists():
            return AgentListResponse(agents=[], count=0)
        
        agents = [d.name for d in agents_dir.iterdir() if d.is_dir()]
        
        return AgentListResponse(
            agents=sorted(agents),
            count=len(agents)
        )
        
    except Exception as e:
        logger.error(f"Error listing agents: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list agents: {str(e)}"
        )

@app.get("/agent/{agent_name}/files", tags=["Agent Management"])
async def get_agent_files(agent_name: str):
    """Get the list of files for a specific agent."""
    try:
        agent_dir = Path(f"generated_agents/{agent_name}")
        if not agent_dir.exists():
            raise HTTPException(status_code=404, detail="Agent not found")
        
        files = []
        for file_path in agent_dir.rglob("*"):
            if file_path.is_file():
                files.append(str(file_path.relative_to(agent_dir)))
        
        return {
            "agent_name": agent_name,
            "files": sorted(files),
            "count": len(files)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting agent files: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get agent files: {str(e)}"
        )

@app.get("/agent/{agent_name}/file/{file_path:path}", tags=["Agent Management"])
async def get_agent_file_content(agent_name: str, file_path: str):
    """Get the content of a specific agent file."""
    try:
        agent_file = Path(f"generated_agents/{agent_name}/{file_path}")
        if not agent_file.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        # Security check - ensure file is within agent directory
        try:
            agent_file.resolve().relative_to(Path(f"generated_agents/{agent_name}").resolve())
        except ValueError:
            raise HTTPException(status_code=403, detail="Access denied")
        
        content = agent_file.read_text(encoding="utf-8")
        
        return {
            "agent_name": agent_name,
            "file_path": file_path,
            "content": content,
            "size": len(content)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting file content: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get file content: {str(e)}"
        )

# Exception handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {str(exc)}")
    logger.error(traceback.format_exc())
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if os.getenv("DEBUG") else "An unexpected error occurred"
        }
    )

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup."""
    logger.info("Starting Google Meta Agent server...")
    logger.info(f"Project ID: {PROJECT_ID}")
    logger.info(f"Region: {REGION}")
    
    # Create necessary directories
    Path("generated_agents").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    
    logger.info("Google Meta Agent server started successfully!")

# Main execution
if __name__ == "__main__":
    import uvicorn
    
    port = int(os.environ.get("PORT", 8080))
    host = os.environ.get("HOST", "0.0.0.0")
    
    logger.info(f"Starting server on {host}:{port}")
    
    uvicorn.run(
        "server:app",
        host=host,
        port=port,
        reload=False,
        workers=1,
        access_log=True
    )
