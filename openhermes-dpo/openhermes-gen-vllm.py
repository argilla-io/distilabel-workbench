"""
git clone https://github.com/argilla-io/distilabel.git
pip install -e ".[vllm]"
"""
import os
from pathlib import Path

from distilabel.llm import vLLM
from distilabel.tasks import TextGenerationTask
from distilabel.pipeline import Pipeline
from datasets import load_dataset
from distilabel.dataset import DatasetCheckpoint

from vllm import LLM

HF_TOKEN = os.getenv("HF_TOKEN")
DATASET_TO_GENERATE = "argilla/OpenHermes-2.5-with-system-2"
MODEL_NAME = "NousResearch/Nous-Hermes-2-Yi-34B"
NEW_DATASET_NAME = "argilla/OpenHermes-2.5-dpo-with-system-ckpt-2"

dataset = load_dataset(DATASET_TO_GENERATE, split="train")

llm = vLLM(
    model=LLM(
        model="NousResearch/Nous-Hermes-2-Yi-34B",
        tensor_parallel_size=4
    ),
    task=TextGenerationTask(),
    prompt_format="chatml",
    max_new_tokens=2048
)

pipeline = Pipeline(generator=llm)

if __name__ == "__main__":

    dataset_checkpoint = DatasetCheckpoint(
        strategy="hf-hub",
        extra_kwargs={"repo_id": NEW_DATASET_NAME, "token": HF_TOKEN},
        save_frequency=10000
    )

    dataset = pipeline.generate(
        dataset,
        num_generations=1,
        checkpoint_strategy=dataset_checkpoint,
        batch_size=512
    )

    dataset.push_to_hub(NEW_DATASET_NAME, token=HF_TOKEN)
