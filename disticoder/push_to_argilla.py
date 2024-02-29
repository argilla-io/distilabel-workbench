"""Create a space in argilla (if doesn't exist) and push the dataset to visualize it.
"""

import argilla as rg
from huggingface_hub import duplicate_space
import time

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--space-name", default="", type=str, required=False)
    
    args = parser.parse_args()
    
    print(args)
    import sys
    sys.exit(0)

    # TODO: Only necessary if creating a new space from scratch
    from_id = "argilla/argilla-template-space"  # default template
    to_id = f"disticoder-dpo-v2"  # New id of the dataset, will reuse the user, otherwise update to your HF account
    new_space = duplicate_space(from_id, to_id=to_id)

    print("Waiting for the space to be created...")
    time.sleep(5 * 60)  # Wait 5 mins for the space to be created

    argilla_api_key = "admin.apikey"
    argilla_space_url = f"https://{new_space.namespace}-{to_id}.hf.space"

    default_workspace = "admin"

    rg.init(
        api_key=argilla_api_key,
        api_url=argilla_space_url,
        workspace=default_workspace
    )

    # TODO: LOAD THE DATASET
    workspace = "admin"

    print(f"Pushing to argilla: '{name}'")
    try:
        dataset_rg = rg.FeedbackDataset.from_argilla(name="code-quality-disticoder-dpo-v2", workspace=workspace)
        dataset_rg.delete()
    except:
        pass
    rg_dataset = datasets[short].to_argilla()
    rg_dataset.push_to_argilla(name="code-quality-disticoder-dpo-v2", workspace=workspace)
