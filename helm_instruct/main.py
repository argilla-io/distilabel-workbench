import os

from criterion import criterion as criterion
from datasets import load_dataset
from distilabel.dataset import DatasetCheckpoint
from distilabel.llm import OpenAILLM
from distilabel.pipeline import Pipeline
from evaluator import HelmInstructTask

dataset = (
    load_dataset("argilla/distilabel-intel-orca-dpo-pairs", split="train")
    .rename_column("input", "prompt")
    .rename_column("chosen", "response")
)
dataset = dataset.select_columns(["prompt", "response"])
dataset = dataset.select(range(10))
OPENAI_API_TOKEN = os.getenv("OPENAI_API_TOKEN")
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
NEW_DATASET_NAME = "argilla/intel-orca-dpo-pairs-helm-instruct"

checkpoint_strategy = DatasetCheckpoint(
    extra_kwargs={
        "repo_id": NEW_DATASET_NAME,
        "token": HF_API_TOKEN,
        "private": True,
        "split": "train",
    },
    save_frequency=500,
)

for criterion_key in criterion:
    pipe = Pipeline(
        labeller=OpenAILLM(
            model="gpt-4-1106-preview",  # gpt-4 turbo
            task=HelmInstructTask(criterion=criterion_key),
            max_new_tokens=512,
            num_threads=8,
            api_key=OPENAI_API_TOKEN,
            temperature=0.3,
        )
    )
    dataset = pipe.generate(
        dataset,
        num_generations=1,
        batch_size=16,
        # checkpoint_strategy=checkpoint_strategy,
    )
    print(dataset)
    dataset.rename_column("generations", f"generations_{criterion_key}")
dataset.push_to_hub(NEW_DATASET_NAME, token=HF_API_TOKEN, use_auth_token=True)
