"""Script to generate the dataset for the DistiCoder-dpo.

The dataset is generated in three steps: sft, dpo, labelling.
The first and last steps were run locally, the second one
was run on runpod with 3 A40 GPUs.

pip install distilabel vllm argilla openai

python oss_instruct_pref.py \
    --step sft
    --hf-apikey ...
    --openai-apikey ...
    --path-snippets "code_snippets.jsonl"
    --save-freq 20

python oss_instruct_pref.py \
    --step dpo \
    --hf-apikey ... \
    --openai-apikey ... \
    --path-snippets "code_snippets.jsonl" \
    --save-freq 20

python ENV/oss_instruct_pref_vllm.py \
    --step label \
    --hf-apikey ...\
    --openai-apikey ... \
    --path-snippets "ENV/code_snippets.jsonl" \
    --save-freq 20 \
    --labelling-task instruction_following

If running on runpod, be sure to add extra space for the container volume and disk
for the models to download properly (200Gb are enough).
"""

import os
import sys
from pathlib import Path

from datasets import Dataset, load_dataset
from distilabel.llm import LLMPool, ProcessLLM
from distilabel.pipeline import Pipeline
from distilabel.tasks import TextGenerationTask
from distilabel.llm import OpenAILLM
from huggingface_hub import login
from distilabel.tasks.preference.ultrafeedback import UltraFeedbackTask, Rating
from distilabel.dataset import CustomDataset

import argilla as rg

import pandas as pd

from typing import Dict, List

from distilabel.tasks import TextGenerationTask, Task
from distilabel.tasks.prompt import Prompt
from distilabel.dataset import DatasetCheckpoint

from dataclasses import dataclass

from distilabel.llm import LLM, LLMPool, ProcessLLM


here = Path(__file__).parent.resolve()

def load_snippets(path: Path = Path("./code_snippets.jsonl")) -> Dataset:
    ds = load_dataset("json", data_files=path, split="train")
    ds = ds.rename_column("snippet", "input")
    return ds


oss_instruct_prompt = """Please gain inspiration from the following random code snippet to create a high-quality programming problem. Present your output in two distinct sections:
[Problem Description] and [Solution].

Code snippet for inspiration:
```
{code}
```

Guidelines for each section:
1. [Problem Description]: This should be **completely self-contained**, providing
all the contextual information one needs to understand and solve the problem.
Assume common programming knowledge, but ensure that any specific context,
variables, or code snippets pertinent to this problem are explicitly included.
2. [Solution]: Offer a comprehensive, **correct** solution that accurately
addresses the [Problem Description] you provided.
"""


@dataclass
class OSSInstruct(TextGenerationTask):
    system_prompt: str = "You are exceptionally skilled at crafting high-quality programming problems and offering precise solutions."

    def generate_prompt(self, input: str) -> Prompt:
        return Prompt(
            system_prompt=self.system_prompt,
            formatted_prompt=oss_instruct_prompt.format(code=input)
          )

    def parse_output(self, output: str) -> List[Dict[str, str]]:
        problem, solution = output.split("[Solution]")
        return {
            "problem": problem.replace("[Problem Description]", "").strip(),
            "solution": solution.strip()
        }


SYSTEM_PROMPT_MAGICODER = "You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions."

MAGICODER_PROMPT = """

@@ Instruction
{instruction}

@@ Response
"""

@dataclass
class MagicoderTask(TextGenerationTask):
    system_prompt: str = SYSTEM_PROMPT_MAGICODER

    def generate_prompt(self, input: str) -> Prompt:
        return Prompt(
            system_prompt=self.system_prompt,
            formatted_prompt=MAGICODER_PROMPT.format(instruction=input)
          )


def load_magicoder(task: Task, cuda_visible_devices: str = "0") -> LLM:
    import os
    from distilabel.llm import vLLM
    from vllm import LLM
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    return vLLM(
        vllm=LLM(
            model="ise-uiuc/Magicoder-S-DS-6.7B",
            trust_remote_code=True,
            tensor_parallel_size=1,
            max_model_len=16384
        ),
        task=task,
        max_new_tokens=1024,
        temperature=1,
    )


def load_openai_gpt35(task):
    return OpenAILLM(
        model="gpt-3.5-turbo",
        task=task,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        max_new_tokens=1024,
        num_threads=4,
        temperature=1
    )


def load_notus(task: Task) -> LLM:
    import os

    from distilabel.llm import vLLM
    from vllm import LLM

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    return vLLM(
        vllm=LLM(
            model="argilla/notus-7b-v1",
            trust_remote_code=True
        ),
        task=task,
        max_new_tokens=1024,
        temperature=1,
        prompt_format="notus",
    )


WIZARDCODER_PROMPT = """

### Instruction:
{instruction}

### Response:
"""


@dataclass
class WizardCoderTask(TextGenerationTask):
    system_prompt: str = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
    def generate_prompt(self, input: str) -> Prompt:
        return Prompt(
            system_prompt=self.system_prompt,
            formatted_prompt=WIZARDCODER_PROMPT.format(instruction=input)
          )


def load_wizardcoder(task: Task, cuda_visible_devices: str = "2") -> LLM:
    import os
    from distilabel.llm import vLLM
    from vllm import LLM
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    return vLLM(
        vllm=LLM(
            model="WizardLM/WizardCoder-15B-V1.0",
            tensor_parallel_size=1,
            trust_remote_code=True
        ),
        task=task,
        max_new_tokens=1024,
        temperature=1,
    )


pool_generation_task = TextGenerationTask(system_prompt=SYSTEM_PROMPT_MAGICODER)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--openai-apikey", type=str, default=None)
    parser.add_argument("--hf-apikey", type=str, default=None, help="Your HuggingFace API key with **WRITE** permission, otherwise it cannot push to hub")
    parser.add_argument("--nrows", type=int, default=-1, help="Number of rows to sample for the dataset")
    parser.add_argument("--push-to-hub", type=bool, default=True, help="Whether to push the dataset to the HuggingFace Hub")
    parser.add_argument("--push-to-argilla", type=bool, default=True, help="Whether to push the dataset to the HuggingFace Hub")
    parser.add_argument("--argilla-apikey", type=str, default="admin.apikey")
    parser.add_argument("--argilla-url", type=str, default="https://plaguss-distilabel-oss-preference.hf.space")
    parser.add_argument("--save-labels", type=str, default="labels.csv", help="Path to save the labels to")
    # 1) Create a new variable to allow generating the dataset up to a point.
    parser.add_argument("--step", type=str, default="labelling", help="Step for the dataset generation. 'sft', 'dpo', 'labelling'")
    parser.add_argument("--path-snippets", type=str, default="code_snippets.jsonl", help="Path to save the labels to")
    parser.add_argument("--save-freq", type=int, default=20, help="Frequency to checkpoint the dataset")
    parser.add_argument("--labelling-task", type=str, default="all", help="Task for the labeller: 'all', 'overall_quality', 'instruction_following', 'honesty', 'truthfulness', 'code_quality'")
    
    args = vars(parser.parse_args())

    # Contains the file from where we extract the code snippets.
    # Start doing the logins to avoid possible failures later on.

    if args["push_to_hub"]:
        login(token=args["hf_apikey"] or os.getenv("HF_API_TOKEN"))
    if args["push_to_argilla"]:
        print("Logging to argilla")
        rg.init(api_key=args["argilla_apikey"], api_url=args["argilla_url"])

    pipe_generation = Pipeline(
        generator=OpenAILLM(
            model="gpt-3.5-turbo",
            task=OSSInstruct(),
            openai_api_key=args["openai_apikey"] or os.getenv("OPENAI_API_KEY"),
            max_new_tokens=1024,
            num_threads=4,
            temperature=1
        )
    )

    print("Reading dataset for the snippets:")
    ds_snippets = load_snippets(path=args["path_snippets"])

    num_generations = len(ds_snippets) if args["nrows"] == -1 else args["nrows"]
    print("num_generations", num_generations)

    save_frequency = len(ds_snippets) // args["save_freq"]

    dataset_checkpoint_sft = DatasetCheckpoint(path=here / "checkpoint_sft", save_frequency=save_frequency)
    dataset_checkpoint_dpo = DatasetCheckpoint(path=here / "checkpoint_dpo", save_frequency=save_frequency)
    print(f"checkpoint paths: {dataset_checkpoint_sft.path} and {dataset_checkpoint_dpo.path}")
    print(f"Save frequency: every {save_frequency} rows.")

    if not dataset_checkpoint_sft.path.is_dir():
        # Generate initial version of the dataset with the OSSInstruct task
        oss_instruct_ds = pipe_generation.generate(
            dataset=ds_snippets,
            num_generations=1,
            batch_size=8,
            checkpoint_strategy=dataset_checkpoint_sft,
        )
    else:
        print("Load the dataset from the checkpoint")
        from distilabel.dataset import CustomDataset
        oss_instruct_ds = CustomDataset.load_from_disk(dataset_checkpoint_sft.path)

    # Some renaming of the variables to prepare it for the next generation step
    oss_instruct_ds = oss_instruct_ds.rename_column("input", "code_snippet")
    oss_instruct_ds = oss_instruct_ds.rename_column("problem", "input")
    oss_instruct_ds = oss_instruct_ds.remove_columns("generations")

    if args["step"] == "sft":
        print("----" * 5)
        print("GENERATED OSS INSTRUCT DATASET, EXITING")
        print("----" * 5)
        sys.exit(0)

    # Prepare the dataset extracting the string from the internal list
    def extract_input(ex):
        if ex["input"]:
            ex["input"] = ex["input"][0]
        return ex

    ds_second = oss_instruct_ds.select_columns(["input"]).map(extract_input)

    # Extra generation step to prepare the dataset for DPO.
    # oss_instruct_ds_second = pipe_generation_2.generate(
    if not dataset_checkpoint_dpo.path.is_dir():
        llm_pool = LLMPool(
            [
                ProcessLLM(task=MagicoderTask(), load_llm_fn=load_magicoder),
                ProcessLLM(task=WizardCoderTask(), load_llm_fn=load_wizardcoder),
                ProcessLLM(task=pool_generation_task, load_llm_fn=load_notus),
            ]
        )

        pipe_generation_pool = Pipeline(generator=llm_pool)

        print("Start generation of the preference dataset")
        oss_instruct_ds_second = pipe_generation_pool.generate(
            dataset=ds_second,
            num_generations=len(llm_pool.llms),
            batch_size=20,
            checkpoint_strategy=dataset_checkpoint_dpo,
        )
    else:
        print("Load the preference dataset from the checkpoint")
        from distilabel.dataset import CustomDataset
        oss_instruct_ds_second = CustomDataset.load_from_disk(dataset_checkpoint_dpo.path)

    if args["step"] == "dpo":
        print("----" * 5)
        print("GENERATED OSS INSTRUCT DATASET FOR DPO, EXITING")
        print("----" * 5)
        sys.exit(0)

    generations = []
    generation_model = []
    generation_prompt = []

    for i, row in enumerate(oss_instruct_ds):
        gen_model = oss_instruct_ds_second[i]["generation_model"] + row["generation_model"] 
        generation_model.append(gen_model)
        gen = (oss_instruct_ds_second[i]["generations"] or [""] * len(llm_pool.llms)) + (row["solution"] or [""])
        generations.append(gen)

    oss_instruct_ds_second = oss_instruct_ds_second.remove_columns("generations")
    oss_instruct_ds_second = oss_instruct_ds_second.add_column("generations", generations)

    # Prepare the dataset for the labelling step.
    ds = Dataset.from_dict(
        {
            "lang": oss_instruct_ds["lang"],
            "input": oss_instruct_ds_second["input"],
            "generation_model": generation_model,
            "generations": generations
        }
    )
 
    ratings = [
        Rating(
            value=1,
            description="**Low Quality**: Code is incorrect, inconsistent, and inefficient.",
        ),
        Rating(
            value=2,
            description="**Moderate Quality**: Code has major errors and inconsistencies, affecting overall functionality.",
        ),
        Rating(
            value=3,
            description="**Good**: Code is partially correct with noticeable issues.",
        ),
        Rating(
            value=4,
            description="**Very Good**: Code is mostly correct and consistent.",
        ),
        Rating(
            value=5,
            description="**Excellent**: Code is entirely correct, consistent, and efficient.",
        ),
    ]

    from textwrap import dedent
    text_description = dedent("""# Code Quality Assessment
    Evaluate the model's generated code based on various criteria:
    1. **Correctness**: Does the code produce the expected output and perform the intended task without errors?
    2. **Maintainability**: Is the code well-structured, easy to understand, and easy to modify?
    3. **Performance**: How well-optimized is the code in terms of runtime performance and resource usage?
    4. **Consistency & Coding Standards**: Does the code follow established coding conventions and maintain a consistent coding style?
    Your role is to provide a holistic assessment considering all the above factors.

    **Scoring**: Rate outputs 1 to 5 based on the overall quality, considering all aspects:
    """)

    uf_code_quality = UltraFeedbackTask(
        system_prompt="Your role is to evaluate code quality based on given criteria.",
        task_description=text_description,
        ratings=ratings,
    )

    labelling_task = args["labelling_task"]
    tasks = {
        "overall_quality": UltraFeedbackTask.for_overall_quality(),
        "instruction_following": UltraFeedbackTask.for_instruction_following(),
        "honesty": UltraFeedbackTask.for_honesty(),
        "truthfulness": UltraFeedbackTask.for_truthfulness(),
        "code_quality": uf_code_quality
    }
    if labelling_task in tasks.keys():
        tasks = {labelling_task: tasks[labelling_task]}
    else:
        raise ValueError(f"The task must be one of: {tasks.keys()}")

    print("Selected the following tasks:", tasks.keys())
    # Create the labeller pipelines with the different tasks

    labeller_pipelines = {}

    for task_name, task in tasks.items():
        labeller_pipelines[task_name] = Pipeline(
            labeller=OpenAILLM(
                model="gpt-4-1106-preview",  # gpt-4 turbo
                task=task,
                max_new_tokens=512,
                num_threads=8,
                openai_api_key=args["openai_apikey"] or os.getenv("OPENAI_API_KEY"),  #os.getenv("OPENAI_API_KEY", None),
                temperature=0.3
            )
        )

    # Generate the datasets for each of the tasks
    datasets = {}

    for task_name, labeller_pipe in labeller_pipelines.items():
        print(f"RUNNING {task_name}")
        checkpoint = DatasetCheckpoint(path=here / f"checkpoint_label_{task_name}", save_frequency=save_frequency)

        if not checkpoint.path.is_dir():

            new_ds = labeller_pipe.generate(
                ds,
                num_generations=1,
                batch_size=8,
                display_progress_bar=True,
                checkpoint_strategy=checkpoint,
            )
            datasets[task_name] = new_ds
        else:
            datasets[task_name] = CustomDataset.load_from_disk(checkpoint.path)

    # Push all the datasets to argilla, delete them if they exist
    dataset_names = [
        ("overall_quality", "overall-quality-preference-oss-instruct"),
        ("instruction_following", "instruction-following-preference-oss-instruct"),
        ("honesty", "honesty-preference-oss-instruct"),
        ("truthfulness", "truthfulness-preference-oss-instruct"),
        ("code_quality", "code-quality-preference-oss-instruct")
    ]
    if not labelling_task == "all":
        dataset_names = [(short, name) for short, name in dataset_names if short == labelling_task]

    workspace = "admin"

    for short, name in dataset_names:
        print(f"Pushing to argilla: '{name}'")
        try:
            dataset_rg = rg.FeedbackDataset.from_argilla(name=name, workspace=workspace)
            dataset_rg.delete()
        except:
            pass
        rg_dataset = datasets[short].to_argilla()
        rg_dataset.push_to_argilla(name=name, workspace=workspace)

    dfs = []

    for task_name, ds in datasets.items():
        dfs.append(pd.Series(ds.to_pandas()["rating"].explode(), name=task_name).reset_index()[task_name])

    df_labels = pd.DataFrame(dfs).T
    df_labels.to_csv(args["save_labels"])
    print("LABEL VALUE_COUNTS", pd.DataFrame(dfs).T.apply(lambda x: x.value_counts()))
