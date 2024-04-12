import os
import json
from tempfile import mktemp
from huggingface_hub import HfApi

hf_api = HfApi()
hub_token = os.getenv("HF_TOKEN")


def push_dataset_to_hub(domain_seed_data, project_name, domain, hub_username):
    repo_id = f"{hub_username}/{project_name}"
    hf_api.create_repo(
        repo_id=repo_id,
        token=hub_token,
        repo_type="dataset",
        exist_ok=True,
    )

    temp_file = mktemp()
    json.dump(domain_seed_data, open(temp_file, "w"))
    hf_api.upload_file(
        path_or_fileobj=temp_file,
        path_in_repo=f"{domain}_seed_data.json",
        token=hub_token,
        repo_id=repo_id,
        repo_type="dataset",
    )

    # create a readme for the project that shows the domain and project name
    readme = f"# {project_name}\n\n## Domain: {domain}"
    perspectives = domain_seed_data.get("perspectives")
    topics = domain_seed_data.get("topics")
    examples = domain_seed_data.get("examples")
    if perspectives:
        readme += "\n\n## Perspectives\n\n"
        for p in perspectives:
            readme += f"- {p}\n"
    if topics:
        readme += "\n\n## Topics\n\n"
        for t in topics:
            readme += f"- {t}\n"
    if examples:
        readme += "\n\n## Examples\n\n"
        for example in examples:
            readme += f"### {example['question']}\n\n{example['answer']}\n\n"

    temp_file = mktemp()

    with open(temp_file, "w") as f:
        f.write(readme)
    hf_api.upload_file(
        path_or_fileobj=temp_file,
        path_in_repo="README.md",
        token=hub_token,
        repo_id=repo_id,
        repo_type="dataset",
    )

    print(f"Dataset uploaded to {repo_id}")
