"""Script to generate responses for DIBT 10K ranked using the SPIN model.

# Uses distilabel==0.6.0
pip install distilabel[vllm]

# First iteration

python generate_iter_spin.py \
    --hf-apikey $HF_API_TOKEN \
    --source-dataset "DIBT/10k_prompts_ranked" \
    --new-dataset "argilla/10k_prompts_ranked_sft" \
    --model-name "teknium/OpenHermes-2.5-Mistral-7B" \
    --batch-size 128 \
    --cuda-devices "0,1"
"""

import os
from dataclasses import dataclass

from datasets import Dataset, load_dataset
from distilabel.dataset import DatasetCheckpoint
from distilabel.llm import LLM
from distilabel.pipeline import Pipeline
from distilabel.tasks import Task, TextGenerationTask
from distilabel.tasks.prompt import Prompt
from huggingface_hub import login


def get_dataset(ds_name: str) -> Dataset:
    if "iter" not in ds_name:
        dataset = load_dataset(ds_name, split="train")
        dataset = dataset.rename_column("prompt", "input")
    else:
        from datasets import concatenate_datasets
        dataset = load_dataset(ds_name)
        dataset = concatenate_datasets([dataset["train"], dataset["test"]])
        def get_input(ex):
            return {"input": ex["real"][0]["content"]}
        dataset = dataset.map(get_input, remove_columns=["real", "generated"])

    return dataset


@dataclass
class SPINTextGenerationTask(TextGenerationTask):
    """Generic task to generate the prompts following SPIN.
    [SPIN](https://github.com/uclaml/SPIN/blob/main/spin/generate.py)
    """
    # This has to be updated for the model to use, it's set to the same used in the
    # original paper
    spin_prompt: str = "### Instruction: {prompt}\n\n### Response:\n"

    def generate_prompt(self, input: str) -> Prompt:
        return Prompt(
            system_prompt=self.system_prompt,
            formatted_prompt=self.spin_prompt.format(prompt=input)
          )


def load_llm(task: Task, cuda_visible_devices: str = "0", model_name: str = "teknium/OpenHermes-2.5-Mistral-7B") -> LLM:
    import os

    from distilabel.llm import vLLM
    from vllm import LLM
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    return vLLM(
        model=LLM(
            model=model_name,
            trust_remote_code=True,
            tensor_parallel_size=len(cuda_visible_devices.split(",")),
        ),
        task=task,
        prompt_format="chatml",
        max_new_tokens=2048,
        top_p=0.95,
        temperature=1,
    )


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

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hf-apikey", type=str, default=None, help="Your HuggingFace API key with **WRITE** permission, otherwise it cannot push to hub"
    )
    parser.add_argument(
        "--source-dataset", type=str, default="DIBT/10k_prompts_ranked", help="Old dataset name"
    )
    parser.add_argument(
        "--real-dataset", type=str, default="argilla/10k_prompts_ranked_with_responses"
    )
    parser.add_argument(
        "--new-dataset", type=str, default="argilla/10k_prompts_ranked_sft", help="New dataset name"
    )
    parser.add_argument(
        "--model-name", type=str, default="teknium/OpenHermes-2.5-Mistral-7B", help="Model to generate the responses"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Model to generate the responses"
    )
    parser.add_argument(
        "--cuda-devices", type=str, default="0,1", help="GPUs to use for vllm, coma separated. Default: 0,1"
    )

    args = parser.parse_args()

    HF_API_TOKEN = args.hf_apikey or os.getenv("HF_API_TOKEN")
    SAVE_FREQ = 500

    login(token=HF_API_TOKEN)

    dataset = get_dataset(args.source_dataset)

    num_generations = len(dataset)
    print("num_generations", num_generations)

    checkpoint = DatasetCheckpoint(
        strategy="hf-hub",
        extra_kwargs={
            "repo_id": args.new_dataset,
            "token": HF_API_TOKEN,
            "private": False,
            "split": "train"
        },
        save_frequency=SAVE_FREQ
    )

    print(f"Save frequency: every {SAVE_FREQ} rows.")

    pipe_generation = Pipeline(
        generator=load_llm(
            SPINTextGenerationTask(),
            model_name=args.model_name,
            cuda_visible_devices=args.cuda_devices
        )
    )

    iterated_dataset = pipe_generation.generate(
        dataset=dataset,
        num_generations=1,
        batch_size=args.batch_size,
        checkpoint_strategy=checkpoint,
    )
    # ---------
    # Load the dataset generated and the reference one to prepare the data
    # for the SPINTrainer
    ds_real = load_dataset(args.real_dataset, split="train")
    ds_generated = load_dataset(args.new_dataset, split="train")

    columns = ["input", "generations"]
    df_real = ds_real.to_pandas()
    df_generated = ds_generated.to_pandas()

    ds_for_spin = Dataset.from_pandas(
        df_generated[columns].merge(
            df_real[columns], on="input"
        ).rename(columns={"generations_x": "generated", "generations_y": "real"}),
        preserve_index=False
    )

    print(args)
    ds_for_spin = ds_for_spin.map(prepare_for_spin, remove_columns=["input"])
    ds_for_spin = ds_for_spin.train_test_split(test_size=0.1, seed=42)
    ds_for_spin.push_to_hub(args.new_dataset, token=HF_API_TOKEN, private=True)

