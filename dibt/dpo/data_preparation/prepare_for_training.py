"""
Script to generate the dataset for DPO fine tuning on DIBT 10K ranked.
Uses the dataset previously generated for SPIN for the chosen/rejected responses
"""

from datasets import load_dataset

def prepare_for_dpo(example):
    return {
        "prompt": [example["real"][0]],
        "chosen": [example["real"][1]],
        "rejected": [example["generated"][1]]
    }


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Prepare the dataset for training using SPIN. Start from the 10k ranked dataset and add the synthetic responses as real.")
    parser.add_argument("--dataset", type=str, default="argilla/10k_prompts_SPIN_iter0_zephyr_top")
    parser.add_argument("--target-dataset", type=str, default="argilla/10k_prompts_dpo")
    
    args = parser.parse_args()

    HF_API_TOKEN = os.getenv("HF_API_TOKEN")
    if not HF_API_TOKEN:
        raise ValueError("You need to set the HF_API_TOKEN environment variable to push the dataset to the hub.")

    ds_base = load_dataset(args.dataset)
    ds_dpo = ds_base.map(prepare_for_dpo, remove_columns=["real", "generated"])
    ds_dpo.push_to_hub(args.target_dataset, token=HF_API_TOKEN, private=True)
