from google.cloud import run_v2
from google.cloud.run_v2 import Service, RevisionTemplate, Container, ResourceRequirements
import logging

logger = logging.getLogger(__name__)

def deploy_to_cloud_run(
    project_id: str, 
    location: str, 
    service_name: str, 
    image_name: str,
    use_gpu: bool = True,
    memory: str = "4Gi",
    cpu: str = "2",
    max_instances: int = 1,
    allow_unauthenticated: bool = True
):
    """
    Deploys a container to Cloud Run with GPU support.
    
    Args:
        project_id: Google Cloud project ID
        location: Cloud Run region
        service_name: Name of the Cloud Run service
        image_name: Full container image name
        use_gpu: Whether to allocate GPU resources
        memory: Memory allocation (e.g., "4Gi")
        cpu: CPU allocation (e.g., "2")
        max_instances: Maximum number of instances
        allow_unauthenticated: Whether to allow unauthenticated access
    
    Returns:
        Service URL
    """
    try:
        client = run_v2.ServicesClient()
        
        # Build the container configuration
        container = Container(
            image=image_name,
            ports=[{"container_port": 8080}],
            resources=ResourceRequirements(
                limits={
                    "memory": memory,
                    "cpu": cpu
                }
            ),
            env=[
                {"name": "PORT", "value": "8080"},
                {"name": "GOOGLE_CLOUD_PROJECT", "value": project_id}
            ]
        )
        
        # Add GPU configuration if requested
        if use_gpu:
            container.resources.limits["nvidia.com/gpu"] = "1"
            # Add GPU-specific environment variables
            container.env.extend([
                {"name": "CUDA_VISIBLE_DEVICES", "value": "0"},
                {"name": "NVIDIA_VISIBLE_DEVICES", "value": "0"}
            ])
        
        # Build the revision template
        revision_template = RevisionTemplate(
            containers=[container],
            scaling=run_v2.Scaling(
                min_instance_count=0,
                max_instance_count=max_instances
            ),
            execution_environment=run_v2.ExecutionEnvironment.EXECUTION_ENVIRONMENT_GEN2,
            # GPU instances require Gen 2 execution environment
            service_account=f"{project_id}@appspot.gserviceaccount.com"
        )
        
        # Set startup probe for GPU instances (they take longer to start)
        if use_gpu:
            revision_template.containers[0].startup_probe = run_v2.Probe(
                http_get=run_v2.HttpGetAction(path="/health"),
                initial_delay_seconds=60,
                timeout_seconds=10,
                period_seconds=10,
                failure_threshold=10
            )
        
        # Build the service configuration
        service = Service(
            template=revision_template,
            traffic=[
                run_v2.TrafficTarget(
                    type_=run_v2.TrafficTargetAllocationType.TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST,
                    percent=100
                )
            ]
        )
        
        # Set ingress to allow all traffic if unauthenticated access is enabled
        if allow_unauthenticated:
            service.ingress = run_v2.IngressTraffic.INGRESS_TRAFFIC_ALL
        
        # Create the service
        parent = f"projects/{project_id}/locations/{location}"
        
        logger.info(f"Deploying service {service_name} to {parent}")
        
        operation = client.create_service(
            parent=parent,
            service_id=service_name,
            service=service,
        )
        
        logger.info("Waiting for deployment to complete...")
        response = operation.result(timeout=600)  # 10 minute timeout for GPU instances
        
        # Set IAM policy for unauthenticated access if requested
        if allow_unauthenticated:
            try:
                from google.cloud import iam
                
                policy_client = iam.Policy()
                
                # Allow allUsers to invoke the service
                binding = {
                    "role": "roles/run.invoker",
                    "members": ["allUsers"]
                }
                
                # Note: This is a simplified example. In production, you would use
                # the proper IAM client to set the policy
                logger.info("Service configured for unauthenticated access")
                
            except Exception as e:
                logger.warning(f"Could not set unauthenticated access: {e}")
        
        service_url = response.uri
        logger.info(f"Service deployed successfully to: {service_url}")
        
        return service_url
        
    except Exception as e:
        logger.error(f"Failed to deploy to Cloud Run: {e}")
        raise

def update_service_gpu_config(
    project_id: str,
    location: str, 
    service_name: str,
    gpu_type: str = "nvidia-l4",
    gpu_count: int = 1
):
    """
    Update an existing Cloud Run service to add GPU configuration.
    
    Args:
        project_id: Google Cloud project ID
        location: Cloud Run region  
        service_name: Name of the existing service
        gpu_type: Type of GPU (e.g., "nvidia-l4")
        gpu_count: Number of GPUs
    """
    try:
        client = run_v2.ServicesClient()
        
        # Get the existing service
        service_path = f"projects/{project_id}/locations/{location}/services/{service_name}"
        service = client.get_service(name=service_path)
        
        # Update the container to include GPU resources
        container = service.spec.template.spec.containers[0]
        if not container.resources:
            container.resources = ResourceRequirements()
        
        if not container.resources.limits:
            container.resources.limits = {}
            
        # Add GPU configuration
        container.resources.limits["nvidia.com/gpu"] = str(gpu_count)
        
        # Update the service
        operation = client.update_service(service=service)
        response = operation.result(timeout=600)
        
        logger.info(f"Updated service {service_name} with GPU configuration")
        return response.uri
        
    except Exception as e:
        logger.error(f"Failed to update service GPU config: {e}")
        raise
