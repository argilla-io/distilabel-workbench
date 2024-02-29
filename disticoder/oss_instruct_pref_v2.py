"""Script to generate the dataset for the DistiCoder-dpo.

The dataset is generated in three steps: sft, dpo, labelling.
The first and last steps were run locally, the second one
was run on runpod with 6 A100 GPUs of 80Gb.

pip install distilabel vllm argilla openai

## ---- Problems step 1 ----
python disticoder/oss_instruct_pref_v2.py \
    --step problems \
    --hf-apikey $HF_API_TOKEN \
    --openai-apikey $OPENAI_API_KEY \
    --push-to-argilla 0 \
    --nrows 5100

## ---- Solutions step 1 ----

python disticoder/oss_instruct_pref_v2.py \
    --step solutions \
    --hf-apikey $HF_API_TOKEN \
    --openai-apikey $OPENAI_API_KEY \
    --push-to-argilla 0 \
    --nrows 5100

## ---- Solutions step 2 ----

python disticoder/oss_instruct_pref_v2.py \
    --step solutions-pool \
    --hf-apikey $HF_API_TOKEN \
    --openai-apikey $OPENAI_API_KEY \
    --push-to-argilla 0 \
    --nrows 5100

## ---- Labelling step 1 ----
TODO: Here we should merge the previous datasets if not found in HF.

python disticoder/oss_instruct_pref_v2.py \
    --step labelling \
    --hf-apikey $HF_API_TOKEN \
    --openai-apikey $OPENAI_API_KEY \
    --labelling-task code_quality \
    --push-to-argilla 0 \
    --nrows 5100 \
    --save-freq 100


If running on runpod, be sure to add extra space for the container volume and disk
for the models to download properly (200Gb are enough).
"""

import os
import re
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


def load_dataset_snippets(seed: int = 422, nrows: int = 5100) -> Dataset:
    ds = load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K", split="train")
    if nrows == -1:
        nrows = len(ds)
    df_sampled = ds.to_pandas().set_index("lang").sample(
        n=nrows,
        random_state=seed
    ).reset_index()
    ds_sampled = Dataset.from_pandas(df_sampled, preserve_index=False)
    return ds_sampled.select_columns(["seed"]).rename_column("seed", "input")


oss_instruct_prompt = """Please gain inspiration from the following random code snippet to create a high-quality programming problem.

Code snippet for inspiration:
```
{code}
```

Guidelines for the problem:
The problem should be **completely self-contained**, providing all the contextual information one needs to understand and solve the problem. The problem must be written as a natural question, avoid titles or anything that would make it artificial. Assume common programming knowledge, but ensure that any specific context, variables, or code snippets pertinent to this problem are **explicitly included*. **Don't reference any provided code snippet** if you are not including it in the problem description."""


@dataclass
class OSSInstructProblem(TextGenerationTask):
    system_prompt: str = "You are exceptionally skilled at crafting high-quality programming problems."

    def generate_prompt(self, input: str) -> Prompt:
        return Prompt(
            system_prompt=self.system_prompt,
            formatted_prompt=oss_instruct_prompt.format(code=input)
          )

    def parse_output(self, output: str) -> List[Dict[str, str]]:
        return {"problem": output}


oss_solution_prompt = """Offer a comprehensive, **correct** solution that accurately addresses the following problem:
{problem}"""


@dataclass
class OSSSolution(TextGenerationTask):
    system_prompt: str = "You are exceptionally skilled at code generation and problem solving."

    def generate_prompt(self, input: str) -> Prompt:
        return Prompt(
            system_prompt=self.system_prompt,
            formatted_prompt=oss_solution_prompt.format(problem=input)
          )

    def parse_output(self, output: str) -> List[Dict[str, str]]:
        return {"solution": output}


MAGICODER_PROMPT = """
@@ Instruction
{instruction}

@@ Response
"""


@dataclass
class MagicoderTask(TextGenerationTask):
    system_prompt: str = OSSSolution().system_prompt

    def generate_prompt(self, input: str) -> Prompt:
        return Prompt(
            system_prompt=self.system_prompt,
            formatted_prompt=MAGICODER_PROMPT.format(instruction=oss_solution_prompt.format(problem=input))
          )

    def parse_output(self, output: str) -> List[Dict[str, str]]:
        return {"solution": output}


def load_magicoder(task: Task, cuda_visible_devices: str = "0") -> LLM:
    import os
    from distilabel.llm import vLLM
    from vllm import LLM
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    return vLLM(
        model=LLM(
            model="ise-uiuc/Magicoder-S-DS-6.7B",
            trust_remote_code=True,
            tensor_parallel_size=len(cuda_visible_devices.split(",")),
            max_model_len=16384
        ),
        task=task,
        max_new_tokens=1024,
        top_p=0.95,
        temperature=1,
    )


def load_notus(task: Task, cuda_visible_devices: str = "1") -> LLM:
    import os

    from distilabel.llm import vLLM
    from vllm import LLM

    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

    return vLLM(
        model=LLM(
            model="argilla/notus-7b-v1",
            trust_remote_code=True,
            tensor_parallel_size=len(cuda_visible_devices.split(",")),
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
    system_prompt: str = OSSSolution().system_prompt
    def generate_prompt(self, input: str) -> Prompt:
        return Prompt(
            system_prompt=self.system_prompt,
            formatted_prompt=WIZARDCODER_PROMPT.format(instruction=oss_solution_prompt.format(problem=input))
          )

    def parse_output(self, output: str) -> List[Dict[str, str]]:
        return {"solution": output}


def load_wizardcoder(task: Task, cuda_visible_devices: str = "2") -> LLM:
    import os
    from distilabel.llm import vLLM
    from vllm import LLM
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    return vLLM(
        model=LLM(
            model="WizardLM/WizardCoder-15B-V1.0",
            tensor_parallel_size=len(cuda_visible_devices.split(",")),
            trust_remote_code=True
        ),
        task=task,
        max_new_tokens=1024,
        temperature=1,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--openai-apikey", type=str, default=None)
    parser.add_argument("--hf-apikey", type=str, default=None, help="Your HuggingFace API key with **WRITE** permission, otherwise it cannot push to hub")
    parser.add_argument("--nrows", type=int, default=-1, help="Number of rows to sample for the dataset")
    parser.add_argument("--push-to-hub", type=bool, default=True, help="Whether to push the dataset to the HuggingFace Hub")
    parser.add_argument("--push-to-argilla", type=int, default=1, help="Whether to push the dataset to the HuggingFace Hub")
    parser.add_argument("--argilla-apikey", type=str, default="admin.apikey")
    parser.add_argument("--argilla-url", type=str, default="https://plaguss-distilabel-oss-preference.hf.space")
    parser.add_argument("--save-labels", type=str, default="labels.csv", help="Path to save the labels to")
    # 1) Create a new variable to allow generating the dataset up to a point.
    parser.add_argument("--step", type=str, default="labelling", help="Step for the dataset generation. 'problems', 'solutions', 'solutions-pool', 'labelling'")
    parser.add_argument("--save-freq", type=int, default=500, help="Frequency to checkpoint the dataset")
    parser.add_argument("--labelling-task", type=str, default="all", help="Task for the labeller: 'all', 'overall_quality', 'instruction_following', 'honesty', 'truthfulness', 'code_quality'")
    parser.add_argument("--seed", type=int, default=422, help="Random seed for the dataset snippets")

    args = parser.parse_args()

    # Contains the file from where we extract the code snippets.
    # Start doing the logins to avoid possible failures later on.
    HF_API_TOKEN = args.hf_apikey or os.getenv("HF_API_TOKEN")
    OPENAI_API_TOKEN = args.openai_apikey or os.getenv("OPENAI_API_KEY")

    if args.push_to_hub:
        login(token=HF_API_TOKEN)
    if args.push_to_argilla:
        print("Logging to argilla")
        rg.init(api_key=args.argilla_apikey, api_url=args.argilla_url)

    print("Reading dataset for the snippets:")
    ds_snippets = load_dataset_snippets(seed=args.seed, nrows=args.nrows)

    num_generations = len(ds_snippets) if args.nrows == -1 else args.nrows
    print("num_generations", num_generations)

    DATASET_NAME_PROBLEMS = "argilla/oss-instruct-problems-step-1"
    DATASET_NAME_SOLUTIONS_OPENAI = "argilla/oss-instruct-solutions-step-1"
    DATASET_NAME_SOLUTIONS_POOL = "argilla/oss-instruct-solutions-step-2"

    dataset_checkpoint_problems = DatasetCheckpoint(
        strategy="hf-hub",
        extra_kwargs={
            "repo_id": DATASET_NAME_PROBLEMS,
            "token": HF_API_TOKEN,
            "private": True,
            "split": "train"
        },
        save_frequency=args.save_freq
    )
    dataset_checkpoint_solutions_openai = DatasetCheckpoint(
        strategy="hf-hub",
        extra_kwargs={
            "repo_id": DATASET_NAME_SOLUTIONS_OPENAI,
            "token": HF_API_TOKEN,
            "private": True,
            "split": "train"
        },
        save_frequency=args.save_freq
    )
    dataset_checkpoint_solutions_pool = DatasetCheckpoint(
        strategy="hf-hub",
        extra_kwargs={
            "repo_id": DATASET_NAME_SOLUTIONS_POOL,
            "token": HF_API_TOKEN,
            "private": True,
            "split": "train"
        },
        save_frequency=args.save_freq
    )

    print(f"Save frequency: every {args.save_freq} rows.")

    try:
        print("Dataset exists, load it from the checkpoint")
        oss_instruct_ds = load_dataset(DATASET_NAME_PROBLEMS, split="train", token=HF_API_TOKEN)
    except Exception as e:
        pipe_generation = Pipeline(
            generator=OpenAILLM(
                model="gpt-3.5-turbo",
                task=OSSInstructProblem(),
                api_key=OPENAI_API_TOKEN,
                max_new_tokens=1024,
                num_threads=8,
                temperature=1
            )
        )
        oss_instruct_ds = pipe_generation.generate(
            dataset=ds_snippets,
            num_generations=1,
            batch_size=16,
            checkpoint_strategy=dataset_checkpoint_problems,
        )

    # Some renaming of the variables to prepare it for the next generation step
    ds_second = (
        oss_instruct_ds
        .rename_column("input", "code_snippet")
        .rename_column("problem", "input")
        .map(lambda ex: {"input": ex["input"][0]})
        .remove_columns(["generations", "generation_model"])
    )

    if args.step == "problems":
        print("----" * 5)
        print("GENERATED OSS INSTRUCT DATASET WITH PROBLEM, EXITING")
        print("----" * 5)
        sys.exit(0)

    if args.step == "solutions":
        print("Start generation of the solutions dataset")
        pipe_generation_solutions = Pipeline(
            generator=OpenAILLM(
                model="gpt-3.5-turbo",
                task=OSSSolution(),
                api_key=OPENAI_API_TOKEN,
                max_new_tokens=1024,
                num_threads=4,
                temperature=1
            )
        )

        oss_instruct_solutions_step_1 = pipe_generation_solutions.generate(
            dataset=ds_second,
            num_generations=1,
            batch_size=16,
            checkpoint_strategy=dataset_checkpoint_solutions_openai,
        )
        print("----" * 5)
        print("Generated OSS Instruct dataset with solutions from OpenAI, exiting")
        print("----" * 5)
        sys.exit(0)

    elif args.step == "solutions-pool":
        print("Start generation of the solutions dataset from LLM Pool")
        llm_pool = LLMPool(
            [
                ProcessLLM(task=MagicoderTask(), load_llm_fn=load_magicoder),
                ProcessLLM(task=WizardCoderTask(), load_llm_fn=load_wizardcoder),
                ProcessLLM(task=OSSSolution(), load_llm_fn=load_notus),
            ]
        )

        pipe_generation_pool = Pipeline(generator=llm_pool)

        print("Start generation of the preference dataset")
        oss_instruct_solutions_step_2 = pipe_generation_pool.generate(
            dataset=ds_second,
            num_generations=len(llm_pool.llms),
            batch_size=64,
            checkpoint_strategy=dataset_checkpoint_solutions_pool,
        )
        print("----" * 5)
        print("Generated OSS Instruct dataset with solutions from an LLM Pool, exiting")
        print("----" * 5)
        sys.exit(0)

    # generations = []
    # generation_model = []
    # generation_prompt = []

    # for i, row in enumerate(oss_instruct_ds):
    #     gen_model = oss_instruct_ds_second[i]["generation_model"] + row["generation_model"] 
    #     generation_model.append(gen_model)
    #     gen = (oss_instruct_ds_second[i]["generations"] or [""] * len(llm_pool.llms)) + (row["solution"] or [""])
    #     generations.append(gen)

    # oss_instruct_ds_second = oss_instruct_ds_second.remove_columns("generations")
    # oss_instruct_ds_second = oss_instruct_ds_second.add_column("generations", generations)

    # # Prepare the dataset for the labelling step.
    # ds = Dataset.from_dict(
    #     {
    #         "lang": oss_instruct_ds["lang"],
    #         "input": oss_instruct_ds_second["input"],
    #         "generation_model": generation_model,
    #         "generations": generations
    #     }
    # )
    # TODO: RELOAD THE MERGED DATASET FROM HF, EXPLAIN HOW IT'S OBTAINED
    # NOTEBOOK review_disticoder.ipynb
    ds = load_dataset("argilla/disticoder-dpo-v2-unlabelled", split="train")
    # Prepare it for the labelling step
    ds = (
        ds.rename_column("problem", "input").rename_column("solutions", "generations")
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

    labelling_task = args.labelling_task
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
                api_key=OPENAI_API_TOKEN,
                temperature=0.3
            )
        )

    # Generate the datasets for each of the tasks
    datasets = {}

    for task_name, labeller_pipe in labeller_pipelines.items():
        print(f"RUNNING {task_name}")
        checkpoint = DatasetCheckpoint(
            strategy="hf-hub",
            extra_kwargs={
                "repo_id": f"argilla/disticoder-dpo-v2-{task_name}",
                "token": HF_API_TOKEN,
                "private": True,
                "split": "train"
            },
            save_frequency=args.save_freq
        )

        new_ds = labeller_pipe.generate(
            ds,
            num_generations=1,
            batch_size=16,
            checkpoint_strategy=checkpoint,
        )
        datasets[task_name] = new_ds

    print("----" * 5)
    print("Labelled OSS Instruct dataset, exiting")
    print("----" * 5)
    sys.exit(0)

    # Push all the datasets to argilla, delete them if they exist
    dataset_names = [
        ("overall_quality", "overall-quality-disticoder-dpo-v2"),
        ("instruction_following", "instruction-following-disticoder-dpo-v2"),
        ("honesty", "honesty-disticoder-dpo-v2"),
        ("truthfulness", "truthfulness-disticoder-dpo-v2"),
        ("code_quality", "code-quality-disticoder-dpo-v2")
    ]
    if not labelling_task == "all":
        dataset_names = [(short, name) for short, name in dataset_names if short == labelling_task]

    # TODO: PLACE THIS INTO A SEPARATE SCRIPT

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
