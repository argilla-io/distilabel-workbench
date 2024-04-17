
from datasets import load_dataset, Dataset, concatenate_datasets
import numpy as np


ds_capy_name = "argilla/distilabel-capybara-dpo-7k-binarized"
ds_orca_name = "argilla/distilabel-intel-orca-dpo-pairs"

ds_capy = load_dataset(ds_capy_name, split="train")
ds_orca = load_dataset(ds_orca_name, split="train")

# Capybara subset

np.random.seed(42)
selection = np.random.randint(1, len(ds_capy), 1000)

cols_capy = ["chosen", "rejected", "rating_chosen", "rating_rejected"]


def prepare_capy(example):
    example["instruction"] = [example["chosen"][0]["content"], example["chosen"][0]["content"]]
    example["response"] = [example["chosen"][1]["content"], example["rejected"][1]["content"]]
    example["rating"] = [example["rating_chosen"], example["rating_rejected"]]
    return example

df_capy = ds_capy.select_columns(cols_capy).select(selection).map(prepare_capy, remove_columns=cols_capy).to_pandas()

df_capy = df_capy.explode(list(df_capy.columns))
df_capy["dataset_name"] = ds_capy_name
df_capy["rating"] = df_capy["rating"].astype(float) 
ds_capy = Dataset.from_pandas(df_capy, preserve_index=False)

# Orca subset

np.random.seed(67)
selection = np.random.randint(1, len(ds_orca), 1000)

cols_orca = ["input", "chosen", "rejected", "rating", "generations"]

def prepare_orca(example):
    example["instruction"] = [example["input"], example["input"]]
    example["response"] = example["generations"]
    return example

df_orca = ds_orca.select_columns(cols_orca).select(selection).map(prepare_orca).select_columns(["instruction", "response", "rating"]).to_pandas()

# Not all the ratings are contained, due to errors in the original generation
df_orca = df_orca[df_orca["rating"].notna()]

df_orca = df_orca.explode(list(df_orca.columns))
df_orca["dataset_name"] = ds_orca_name
ds_orca = Dataset.from_pandas(df_orca, preserve_index=False)

# Concatenate both datasets
new_ds = concatenate_datasets([ds_capy, ds_orca], axis=0).shuffle(42)

new_ds.push_to_hub("distilabel-internal-testing/critique-bench-dpo-mix-4k", split="train")
