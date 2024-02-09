import json
from math import e
from textwrap import dedent
from typing import Any, Dict

from datasets import Dataset
from distilabel.llm import JSONOpenAILLM
from distilabel.pipeline import Pipeline
from distilabel.tasks import Prompt, TextGenerationTask
from dotenv import load_dotenv

from src.examples import example_tools, example_function_responses, FunctionCallResponse
from src.utils import filter_column_not_none

load_dotenv("../.env")


class FunctionResponseGeneratorTask(TextGenerationTask):
    example_functions = example_tools
    example_function_responses = example_function_responses

    @property
    def input_args_names(self):
        return ["instruction", "function"]

    @property
    def output_args_names(self):
        return ["function_call", "function_call_parses", "function_call_models"]

    def parse_output(self, output: str) -> Dict[str, Any]:
        try:
            output = json.loads(output)
        except json.JSONDecodeError:
            output = literal_eval(output)
        parses = True
        try:
            function_call = FunctionCallResponse(output).model_dump_json()
            models = True
        except Exception as e:
            function_call = str(output)
            models = False
        return {
            "function_call_parses": parses,
            "function_call_models": models,
            "function_call": function_call,
        }

    def generate_prompt(self, instruction: str, function: str, **_: Any):
        formatted_prompt = f"Instruction: {instruction}\n"
        formatted_prompt += f"\n\nFunction: {function}\n"
        example_schema = self.example_function_responses[0].model_json_schema()
        formatted_prompt += (
            f"The schema for you to call the function call is : {example_schema}\n"
        )
        for example in self.example_function_responses:
            formatted_prompt += f"\n\nExample function calls: {example.model_dump()}\n"
        return Prompt(
            system_prompt=self.system_prompt, formatted_prompt=formatted_prompt
        )


task = FunctionResponseGeneratorTask(
    system_prompt=(
        "You are an AI assistant that that performs tasks using functions."
        "You are are given an instruction and a function."
        "To use the function, you must respond with a JSON in the correct format."
    )
)
function_response_generator = JSONOpenAILLM(
    task=task,
    model="gpt-4-1106-preview",
)


def generate(
    dataset: Dataset,
    num_generations: int = 2,
    batch_size: int = 5,
    checkpoint_strategy=None,
    max_inputs: int = None,
):

    pipeline = Pipeline(generator=function_response_generator)
    dataset = Dataset.from_list(dataset.to_list()[:max_inputs])
    _response_dataset = pipeline.generate(
        dataset=dataset,
        num_generations=num_generations,
        batch_size=batch_size,
        checkpoint_strategy=checkpoint_strategy,
    )

    response_dataset = unwrap(_response_dataset)
    return response_dataset


def unwrap(dataset):
    df = dataset.to_pandas().reset_index()
    df = df.loc[:, ~df.columns.str.contains("_index_")]
    df = df.explode(task.output_args_names)
    dataset = Dataset.from_pandas(df)
    dataset = filter_column_not_none(dataset, "function_call")
    return dataset
