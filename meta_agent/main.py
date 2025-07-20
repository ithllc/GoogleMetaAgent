import os
from meta_agent.agent_generation.generator import AgentGenerator
from meta_agent.agent_generation.templating import create_agent_files
from meta_agent.vllm_config.dockerfile_generator import generate_dockerfile
from meta_agent.deployment.cloud_build import trigger_cloud_build
from meta_agent.deployment.cloud_run import deploy_to_cloud_run

def main():
    # 1. Get user description
    description = "A simple RAG agent that uses Google Search to answer questions."

    # 2. Generate agent code
    agent_generator = AgentGenerator()
    agent_blueprint = agent_generator.forward(description)
    agent_files = create_agent_files(agent_blueprint)

    # 3. Create a directory for the new agent
    agent_dir = f"generated_agents/{agent_blueprint.agent_name}"
    os.makedirs(agent_dir, exist_ok=True)

    # 4. Write the agent files
    for file_name, content in agent_files.items():
        with open(os.path.join(agent_dir, file_name), "w") as f:
            f.write(content)

    # 5. Generate Dockerfile
    dockerfile = generate_dockerfile(agent_blueprint.agent_name, ["dspy-ai", "google-cloud-aiplatform"])
    with open(os.path.join(agent_dir, "Dockerfile"), "w") as f:
        f.write(dockerfile)

    # 6. Generate requirements.txt
    with open(os.path.join(agent_dir, "requirements.txt"), "w") as f:
        f.write("dspy-ai\n")
        f.write("google-cloud-aiplatform\n")

    # 7. Build and deploy
    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
    gcs_source = f"gs://{project_id}-meta-agent-source"
    image_name = f"gcr.io/{project_id}/{agent_blueprint.agent_name}"
    location = "us-central1"

    # In a real implementation, you would upload the source to GCS
    # For now, we'll just print the commands
    print(f"gsutil mb {gcs_source}")
    print(f"tar -czf source.tar.gz {agent_dir}")
    print(f"gsutil cp source.tar.gz {gcs_source}/source.tar.gz")

    # trigger_cloud_build(project_id, gcs_source, image_name)
    # deploy_to_cloud_run(project_id, location, agent_blueprint.agent_name, image_name)

if __name__ == "__main__":
    main()
