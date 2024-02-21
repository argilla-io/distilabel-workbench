import json
from ast import literal_eval
from typing import Any, Dict

from datasets import Dataset
from distilabel.llm import JSONOpenAILLM, OpenAILLM
from distilabel.pipeline import Pipeline
from distilabel.tasks import Prompt, TextGenerationTask
from dotenv import load_dotenv

from src.examples import (
    example_tools,
    example_function_responses,
    FunctionCallResponseArray,
)
from src.utils import filter_column_not_none

load_dotenv("../.env")


class FunctionResponseGeneratorTask(TextGenerationTask):
    example_functions = example_tools
    example_function_responses = example_function_responses

    __jinja2_template__ = "/home/ben/code/distilabel-workbench/scripts/function_calling_dataset/templates/functionresponses.jinja2"

    @property
    def input_args_names(self):
        return ["instructions", "function"]

    def parse_output(self, output: str) -> Dict[str, Any]:
        try:
            output = json.loads(output)
        except json.JSONDecodeError:
            output = literal_eval(output)
        try:
            function_calls = FunctionCallResponseArray(**output).function_calls
            function_calls = [model.model_dump_json() for model in function_calls]
        except Exception:
            function_calls = [json.dumps(obj) for obj in output]
        return {
            "generations": function_calls,
        }

    def generate_prompt(self, instructions: str, function: str, **_: Any):
        example_schema = self.example_function_responses.model_json_schema()
        example_function_calls = [
            example.model_dump_json()
            for example in self.example_function_responses.function_calls
        ]
        render_kwargs = {
            "function": function,
            "schema": example_schema,
            "instructions": instructions,
            "examples": example_function_calls,
        }
        return Prompt(
            system_prompt=self.system_prompt,
            formatted_prompt=self.template.render(**render_kwargs),
        )


class TextGenerationTask(TextGenerationTask):

    __jinja2_template__ = "/home/ben/code/distilabel-workbench/scripts/function_calling_dataset/templates/instructions.jinja2"

    def generate_prompt(self, input: str, **_: Any) -> Prompt:
        system_prompt = self.system_prompt
        if self.principles_distribution is not None:
            principle = self._get_principle()
            system_prompt += " " + principle
        render_kwargs = {"input": input}
        return Prompt(
            system_prompt=system_prompt,
            formatted_prompt=self.template.render(**render_kwargs),
        )


call_task = FunctionResponseGeneratorTask(
    system_prompt=(
        "You are an AI assistant that that performs tasks using functions."
        "You are are given an instruction and a function."
        "To use the function, you must respond with a JSON in the correct format."
    )
)

non_call_task = TextGenerationTask()

llm_map = {
    "function_call": JSONOpenAILLM(
        task=call_task,
        model="gpt-4-1106-preview",
        max_new_tokens=4096,
    ),
    "non_calls": OpenAILLM(
        task=non_call_task,
        model="gpt-4",
    ),
}


def generate(
    dataset: Dataset,
    num_generations: int = 2,
    batch_size: int = 5,
    checkpoint_strategy=None,
    max_inputs: int = None,
    task_name: str = "function_call",
):
    """Generate function call responses using GPT-4."""
    llm = llm_map[task_name]
    pipeline = Pipeline(generator=llm)
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
    dataset = Dataset.from_pandas(df)
    dataset = filter_column_not_none(dataset, "generations")
    return dataset


def unwrap_expansions(dataset, call: bool = False):
    df = dataset.to_pandas()
    df = df.loc[:, ~df.columns.str.contains("_index_")]
    df["generations"] = df["generations"].apply(
        lambda response: [json.dumps({"text": r}) for r in response]
    )
    df["call"] = call
    df = df.rename(columns={"input": "instructions"})
    dataset = Dataset.from_pandas(df)
    dataset = filter_column_not_none(dataset, "generations")
    return dataset
