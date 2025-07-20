from google.cloud import run_v2

def deploy_to_cloud_run(project_id: str, location: str, service_name: str, image_name: str):
    """
    Deploys a container to Cloud Run.
    """
    client = run_v2.ServicesClient()

    service = {
        "template": {
            "containers": [{"image": image_name}],
            "scaling": {"min_instance_count": 1, "max_instance_count": 1},
            "resources": {"cpu": 1, "memory": "4Gi", "startup_cpu_boost": True, "gpus": 1},
        }
    }

    operation = client.create_service(
        parent=f"projects/{project_id}/locations/{location}",
        service_id=service_name,
        service=service,
    )

    print("Waiting for service to be ready...")
    response = operation.result()
    print(f"Service deployed to: {response.uri}")
    return response.uri
