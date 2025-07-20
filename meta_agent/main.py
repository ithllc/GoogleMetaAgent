import os
import logging
from pathlib import Path
from meta_agent.agent_generation.generator import AgentGenerator
from meta_agent.agent_generation.templating import create_agent_files
from meta_agent.vllm_config.dockerfile_generator import (
    generate_dockerfile, 
    generate_requirements_txt, 
    generate_cloudbuild_yaml
)
from meta_agent.vllm_config.vllm_instance_manager import VLLMInstanceManager
from meta_agent.deployment.cloud_build import trigger_cloud_build
from meta_agent.deployment.cloud_run import deploy_to_cloud_run

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MetaAgent:
    """
    Meta Agent that generates, builds, and deploys other agents.
    
    This implements the core ADAS (Automated Design of Agentic Systems) concept
    where a meta agent automatically creates new agents based on natural language descriptions.
    """
    
    def __init__(self, project_id: str = None, location: str = "us-central1"):
        self.project_id = project_id or os.environ.get("GOOGLE_CLOUD_PROJECT")
        self.location = location
        self.agent_generator = AgentGenerator()
        
        if not self.project_id:
            raise ValueError("Google Cloud project ID must be provided via project_id parameter or GOOGLE_CLOUD_PROJECT environment variable")
    
    def create_and_deploy_agent(self, description: str, agent_output_dir: str = None) -> str:
        """
        Complete end-to-end agent creation and deployment pipeline.
        
        This implements the workflow described in the MetaAgent Requirements:
        get_description -> generate_agent_code -> build_container -> deploy_to_cloud_run -> return_url
        
        Args:
            description: Natural language description of the desired agent
            agent_output_dir: Directory to store generated agent files (optional)
            
        Returns:
            URL of the deployed agent
        """
        try:
            logger.info(f"Starting meta agent pipeline for: {description}")
            
            # Phase 2, Task 4: Agent Generation Module
            logger.info("Generating agent blueprint...")
            agent_blueprint = self.agent_generator.forward(description)
            logger.info(f"Generated agent: {agent_blueprint.agent_name}")
            
            # Create output directory
            if agent_output_dir is None:
                agent_output_dir = f"generated_agents/{agent_blueprint.agent_name}"
            
            agent_dir = Path(agent_output_dir)
            agent_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Using output directory: {agent_dir}")
            
            # Generate agent files
            logger.info("Creating agent files...")
            agent_files = create_agent_files(agent_blueprint)
            
            # Write agent files to directory
            for file_name, content in agent_files.items():
                file_path = agent_dir / file_name
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)
                logger.info(f"Created: {file_path}")
            
            # Phase 2, Task 5: VLLM Configuration and Containerization Module
            logger.info("Generating deployment files...")
            
            # Generate requirements.txt
            requirements_content = generate_requirements_txt(
                base_requirements=["dspy-ai", "google-cloud-aiplatform"],
                include_vllm=True,
                include_adk=True
            )
            with open(agent_dir / "requirements.txt", "w") as f:
                f.write(requirements_content)
            logger.info("Created: requirements.txt")
            
            # Generate Dockerfile
            dockerfile_content = generate_dockerfile(
                agent_blueprint.agent_name, 
                requirements_content.split("\\n"),
                use_gpu=True
            )
            with open(agent_dir / "Dockerfile", "w") as f:
                f.write(dockerfile_content)
            logger.info("Created: Dockerfile")
            
            # Generate Cloud Build YAML
            cloudbuild_content = generate_cloudbuild_yaml(
                self.project_id,
                agent_blueprint.agent_name,
                self.location
            )
            with open(agent_dir / "cloudbuild.yaml", "w") as f:
                f.write(cloudbuild_content)
            logger.info("Created: cloudbuild.yaml")
            
            # Phase 2, Task 6: Deployment Orchestration Module
            logger.info("Starting deployment process...")
            
            # For now, we'll generate the commands but not execute them automatically
            # In a production environment, you would uncomment the deployment steps
            
            gcs_source_bucket = f"{self.project_id}-meta-agent-source"
            image_name = f"gcr.io/{self.project_id}/{agent_blueprint.agent_name}"
            
            # Print deployment commands for manual execution
            print("\\n" + "="*60)
            print("DEPLOYMENT COMMANDS")
            print("="*60)
            print("Run the following commands to deploy your agent:")
            print()
            print(f"# 1. Create GCS bucket (if it doesn't exist)")
            print(f"gsutil mb gs://{gcs_source_bucket} || true")
            print()
            print(f"# 2. Upload source code")
            print(f"cd {agent_dir}")
            print(f"tar -czf source.tar.gz .")
            print(f"gsutil cp source.tar.gz gs://{gcs_source_bucket}/")
            print()
            print(f"# 3. Trigger Cloud Build")
            print(f"gcloud builds submit --config cloudbuild.yaml .")
            print()
            print(f"# 4. Get the deployed service URL")
            print(f"gcloud run services describe {agent_blueprint.agent_name} --region={self.location} --format='value(status.url)'")
            print()
            
            # Commented out for safety - uncomment to enable automatic deployment
            # logger.info("Triggering Cloud Build...")
            # build_result = trigger_cloud_build(self.project_id, gcs_source_bucket, image_name)
            # 
            # logger.info("Deploying to Cloud Run...")
            # service_url = deploy_to_cloud_run(
            #     self.project_id, 
            #     self.location, 
            #     agent_blueprint.agent_name, 
            #     image_name
            # )
            
            # For now, return a placeholder URL
            service_url = f"https://{agent_blueprint.agent_name}-{self.project_id}.{self.location}.run.app"
            
            logger.info(f"Meta agent pipeline completed successfully!")
            logger.info(f"Generated agent files in: {agent_dir}")
            logger.info(f"Predicted service URL: {service_url}")
            
            return service_url
            
        except Exception as e:
            logger.error(f"Error in meta agent pipeline: {str(e)}")
            raise
    
    def run_local_vllm_instance(self, model_paths: dict, debug_mode: bool = True):
        """
        Run a local VLLM instance for testing purposes.
        
        Args:
            model_paths: Dictionary mapping model names to paths
            debug_mode: Enable debug logging
        """
        logger.info("Starting local VLLM instance for testing...")
        
        vllm_manager = VLLMInstanceManager(
            model_paths=model_paths,
            debug_mode=debug_mode
        )
        
        success = vllm_manager.start_instances()
        if success:
            logger.info("VLLM instances started successfully")
            return vllm_manager
        else:
            logger.error("Failed to start VLLM instances")
            return None


def main():
    """
    Main entry point demonstrating the Meta Agent capabilities.
    """
    # Example usage of the Meta Agent
    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not project_id:
        print("Please set GOOGLE_CLOUD_PROJECT environment variable")
        return
    
    # Initialize Meta Agent
    meta_agent = MetaAgent(project_id=project_id)
    
    # Example agent descriptions
    example_descriptions = [
        "A simple RAG agent that uses Google Search to answer questions about current events",
        "A math agent that can solve equations and perform calculations using code execution",
        "A multi-tool agent that can search the web, execute code, and help with research tasks"
    ]
    
    # Generate agents for each description
    for i, description in enumerate(example_descriptions, 1):
        print(f"\\n{'='*80}")
        print(f"EXAMPLE {i}: {description}")
        print('='*80)
        
        try:
            service_url = meta_agent.create_and_deploy_agent(description)
            print(f"✅ Agent generated successfully!")
            print(f"🔗 Service URL: {service_url}")
        except Exception as e:
            print(f"❌ Error generating agent: {e}")
    
    print(f"\\n{'='*80}")
    print("META AGENT DEMO COMPLETED")
    print('='*80)
    print("Check the 'generated_agents' directory for the created agent files.")
    print("Follow the deployment commands shown above to deploy to Google Cloud Run.")


if __name__ == "__main__":
    main()
