from google.cloud.devtools import cloudbuild_v1

def trigger_cloud_build(project_id: str, gcs_source: str, image_name: str):
    """
    Triggers a Cloud Build job.
    """
    client = cloudbuild_v1.CloudBuildClient()

    build = cloudbuild_v1.Build()
    build.source = {"storage_source": {"bucket": gcs_source, "object_": "source.tar.gz"}}
    build.steps = [
        {
            "name": "gcr.io/cloud-builders/docker",
            "args": ["build", "-t", image_name, "."],
        }
    ]
    build.images = [image_name]

    operation = client.create_build(project_id=project_id, build=build)
    print("Waiting for build to complete...")
    result = operation.result()
    print(result)
