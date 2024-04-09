"""
Scripts to clean and prepare the Prometheus datasets as benchmarks.
"""

import re
import pandas as pd
from datasets import Dataset

DATASETS = {
    "mt_bench_eval": r"https://raw.githubusercontent.com/kaistAI/prometheus/main/evaluation/benchmark/data/mt_bench_eval.json",
    "vicuna_eval": r"https://raw.githubusercontent.com/kaistAI/prometheus/main/evaluation/benchmark/data/vicuna_eval.json",
    "flask_eval": r"https://raw.githubusercontent.com/kaistAI/prometheus/main/evaluation/benchmark/data/flask_eval.json",
    "feedback_eval": r"https://raw.githubusercontent.com/kaistAI/prometheus/main/evaluation/benchmark/data/feedback_collection_ood_test.json"
}

def extract_instruction_response(example):
    pat1 = r'###The instruction to evaluate:(.*?)###Response to evaluate:'
    pat2 = r'###Response to evaluate:(.*?)###Reference Answer'
    matches_1 = re.search(pat1, example["instruction"], re.DOTALL)
    matches_2 = re.search(pat2, example["instruction"], re.DOTALL)
    instruction = response = None
    if matches_1:
        instruction = matches_1.group(1).strip()
    if matches_2:
        response = matches_2.group(1).strip()
    return {
        "instruction": instruction,
        "response": response
    }


if __name__ == "__main__":
    dfs = []
    for ds_name, filename in DATASETS.items():
        if ds_name in {"mt_bench_eval", "vicuna_eval"}:
            reader = lambda filename: pd.read_json(filename, lines=True)
        else:
            reader = lambda filename: pd.read_json(filename, orient="records")
        print(f"Reading dataset: {ds_name}")
        df = reader(DATASETS[ds_name])
        df["base_dataset"] = ds_name
        dfs.append(df)

    df = pd.concat(dfs)

    # Remove incomplete columns, we need to recompute those ourselves attending to our prompt format and model
    df = df.drop(columns=["gpt4_score", "gpt4_feedback", "human_score"])

    ds = Dataset.from_pandas(df, preserve_index=False).map(extract_instruction_response)
    ds.push_to_hub("distilabel-internal-testing/prometheus-bench-critique", split="test")
