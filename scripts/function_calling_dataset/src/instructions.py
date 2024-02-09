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


def generate_instructions(
    dataset: Dataset,
    num_generations: int = 4,
    batch_size: int = 5,
):
    task = FunctionInstructionTask(
        application_description=(
            "A question-answering assistant for productivity that uses function calls to perform tasks."
            "For example, if you are given a function to collect the weather report, you can answer questions about the weather."
        ),
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
    instruction_generator = OpenAILLM(
        task=task,
        model="gpt-4-1106-preview",
    )
    pipeline = Pipeline(generator=instruction_generator)
    dataset = dataset.rename_column("function", "input")
    _instructions_dataset = pipeline.generate(
        dataset=dataset,
        num_generations=num_generations,
        batch_size=batch_size,
        checkpoint_strategy=None,
    )

    def unwrap_functions_instructions(dataset):
        df = dataset.to_pandas()
        df = df.explode("instructions").explode("instructions")
        df = df.explode("input")
        df = df.rename(columns={"input": "function", "instructions": "instruction"})
        # drop any column with _index_ in the name
        df = df.loc[:, ~df.columns.str.contains("_index_")]
        return Dataset.from_pandas(df)

    instructions_dataset = unwrap_functions_instructions(_instructions_dataset)
    instructions_dataset = instructions_dataset.rename_column(
        "instructions", "instruction"
    )
    instructions_dataset = filter_column_not_none(instructions_dataset, "instruction")
    return instructions_dataset
