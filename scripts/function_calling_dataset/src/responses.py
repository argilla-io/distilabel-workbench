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
        return ["function_call"]

    def parse_output(self, output: str) -> Dict[str, Any]:
        function_call = FunctionCallResponse(**json.loads(output)).model_dump_json()
        return {"function_call": function_call}

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


def generate_responses(
    dataset: Dataset,
    num_generations: int = 2,
    batch_size: int = 5,
):
    function_response_generator = JSONOpenAILLM(
        task=FunctionResponseGeneratorTask(
            system_prompt=(
                "You are an AI assistant that that performs tasks using functions."
                "You are are given an instruction and a function."
                "To use the function, you must respond with a JSON in the correct format."
            )
        ),
        model="gpt-4-1106-preview",
    )

    pipeline = Pipeline(generator=function_response_generator)

    _response_dataset = pipeline.generate(
        dataset=dataset,
        num_generations=num_generations,
        batch_size=batch_size,
        checkpoint_strategy=None,
    )

    def unwrap_function_responses_dataset(dataset):
        df = dataset.to_pandas().drop(columns=["__index_level_0__"]).reset_index()
        df = df.explode("function_call")
        return Dataset.from_pandas(df)

    response_dataset = unwrap_function_responses_dataset(_response_dataset)
    response_dataset = filter_column_not_none(response_dataset, "function_call")
    return response_dataset
