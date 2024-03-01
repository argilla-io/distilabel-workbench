"""
Script to generate the initial SPIN iter0 dataset for DIBT 10K ranked.
"""

from datasets import load_dataset, concatenate_datasets

def prepare_for_spin(example):
    return {
        "real": [
            {"role": "user", "content": example["input"]}, 
            {"role": "assistant", "content": example["real"][0]}
        ],
        "generated": [
            {"role": "user", "content": example["input"]}, 
            {"role": "assistant", "content": example["generated"][0]}
        ]
    }


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Prepare the dataset for training using SPIN. Start from the 10k ranked dataset and add the synthetic responses as real.")
    parser.add_argument("--real-dataset", type=str, default="argilla/10k_prompts_ranked_with_responses")
    parser.add_argument("--sft-dataset", type=str, default="argilla/10k_prompts_ranked_sft")
    parser.add_argument("--target-dataset", type=str, default="argilla/10k_prompts_SPIN_iter0")
    parser.add_argument("--portion", type=str, default=1)
    # 1) Create a new variable to allow generating the dataset up to a point.
    
    args = parser.parse_args()

    HF_API_TOKEN = os.getenv("HF_API_TOKEN")
    if not HF_API_TOKEN:
        raise ValueError("You need to set the HF_API_TOKEN environment variable to push the dataset to the hub.")

    ds_real = load_dataset(args.real_dataset, split="train")
    ds_sft = load_dataset(args.sft_dataset, split="train")

    if args.portion == "top":
        df_real = ds_real.to_pandas()

        indices = df_real[
            (df_real["num_responses"] > 1) &
            (df_real["avg_rating"] >= 4)
        ].index

        ds_real = ds_real.select(indices)
        ds_sft = ds_sft.select(indices)
        if "top" not in args.target_dataset:
            args.target_dataset += "_bottom"

    elif args.portion == "bottom":
        df_real = ds_real.to_pandas()

        indices = df_real.sort_values(by="avg_rating", ascending=False).iloc[-2000:].index

        ds_real = ds_real.select(indices)
        ds_sft = ds_sft.select(indices)
        if "bottom" not in args.target_dataset:
            args.target_dataset += "_bottom"

    elif args.portion == "all":
        pass

    else:
        raise ValueError("Available dataset portions are : 'top', 'bottom', 'all'.")

    ds_for_spin = concatenate_datasets(
        [
            ds_real.select_columns(["input", "generations"]).rename_column("generations", "real"),
            ds_sft.select_columns("generations").rename_column("generations", "generated"),
        ],
        axis=1
    )
    print(args)
    ds_for_spin = ds_for_spin.map(prepare_for_spin, remove_columns=["input"])
    ds_for_spin.push_to_hub(args.target_dataset, token=HF_API_TOKEN, split="train", private=True)
