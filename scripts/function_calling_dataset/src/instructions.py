import json
from typing import Dict, List

from datasets import Dataset
from distilabel.llm import OpenAILLM
from distilabel.pipeline import Pipeline
from distilabel.tasks import SelfInstructTask
from dotenv import load_dotenv

from src.utils import filter_column_not_none

load_dotenv("../.env")


class FunctionInstructionTask(SelfInstructTask):

    def parse_output(self, output: str) -> Dict[str, List[str]]:
        """Parses the output of the model into the desired format."""
        instructions = output.split("\n")
        instructions = filter(lambda x: x is not None, instructions)
        instructions = filter(lambda i: len(i) > 5, instructions)
        instructions = map(str.strip, instructions)
        return {"instructions": list(instructions)}


application_description = (
    "A question-answering assistant for productivity that uses function calls to perform tasks."
    "For example, if you are given a function to collect the weather report, you can answer questions about the weather."
)

call_task = FunctionInstructionTask(
    application_description=application_description,
    criteria_for_query_generation=(
        "The instruction should require the function to be answered."
        "The instruction should not specifically mention the function by name."
        "The instruction should include all parameters required for the function."
        "For example: "
        "Weather function >> 'What is the weather like today in London?'"
        "Search function >> 'What was the score of the last el classico game?'"
        "Web scraping function >> 'Can you collect some data from www.newslink.com?'"
    ),
)

non_call_task = FunctionInstructionTask(
    application_description=application_description
    + (
        "You are also able to answer queries that do not require function."
        "For example, if you are asked to answer a questions that require general knowledge."
        "You can answer questions like 'What is the capital of France?'"
        "For example: if you are asked to answer a question that require reasoning."
        "You can answer questions like 'How should I invest if I want to retire early?'"
    ),
    criteria_for_query_generation=(
        "The instruction should not require the function to be answered."
        "The instruction should not specifically mention the function by name."
        "The instruction should not include parameters for the function."
        "For example: "
        "General knowledge >> 'What is the capital of France?'"
        "Reasoning >> 'How should I invest if I want to retire early?'"
    ),
)

task_map = {
    "call": call_task,
    "non_calls": non_call_task,
}


def generate(
    dataset: Dataset,
    num_generations: int = 4,
    batch_size: int = 5,
    checkpoint_strategy=None,
    max_inputs: int = None,
    task_name: str = "call",
):
    """Generate function instructions using GPT-4."""
    task = task_map[task_name]
    instruction_generator = OpenAILLM(
        task=task,
        model="gpt-4-1106-preview",
    )
    pipeline = Pipeline(generator=instruction_generator)
    dataset = dataset.rename_column("function", "input")
    dataset = Dataset.from_list(dataset.to_list()[:max_inputs])
    instructions_dataset = pipeline.generate(
        dataset=dataset,
        num_generations=num_generations,
        batch_size=batch_size,
        checkpoint_strategy=checkpoint_strategy,
    )
    return instructions_dataset


def unwrap(dataset):
    df = dataset.to_pandas()
    df = df.explode("instructions")
    df = df.explode("input")
    df = df.rename(columns={"input": "function"})
    # drop any column with _index_ in the name
    df = df.loc[:, ~df.columns.str.contains("_index_")]
    dataset = Dataset.from_pandas(df)
    dataset = filter_column_not_none(dataset, "instructions")
    return dataset


def unwrap_expansions(dataset, call: bool = False):
    """Wrangle the dataset for the expansion phase"""
    df = dataset.to_pandas()
    df["call"] = call
    df = df.drop(columns=["instructions"])
    df = df.rename(columns={"input": "function"})
    df = df.explode("instructions")
    df = df.rename(columns={"instructions": "input"})
    # drop any column with _index_ in the name
    df = df.loc[:, ~df.columns.str.contains("_index_")]
    dataset = Dataset.from_pandas(df)
    dataset = filter_column_not_none(dataset, "input")
    return dataset
