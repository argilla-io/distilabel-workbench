from ast import literal_eval
import json
from typing import Any, Dict

from datasets import Dataset
from distilabel.llm import JSONOpenAILLM
from distilabel.pipeline import Pipeline
from distilabel.tasks import Prompt, TextGenerationTask
from dotenv import load_dotenv

from src.examples import example_tools, example_function_domain, Tool
from src.utils import filter_column_not_none

load_dotenv("../.env")


class FunctionGeneratorTask(TextGenerationTask):
    system_prompt = (
        "You are a JSON schema generator that responds only using json structures,"
        "You are asked to generate a json schema for a function that performs a specific task."
    )

    examples = example_tools

    @property
    def output_args_names(self):
        return ["generations", "parses", "models"]

    def parse_output(self, output: str) -> Dict[str, Any]:
        try:
            output = json.loads(output)
        except json.JSONDecodeError:
            output = literal_eval(output)
        parses = True
        try:
            output = Tool(**output).model_dump_json()
            models = True
        except Exception as e:
            output = str(output)
            models = False
        return {
            "generations": output,
            "parses": parses,
            "models": models,
        }

    def generate_prompt(self, input: str, **_):
        system_prompt = self.system_prompt
        example = self.examples[0]
        system_prompt += f"\n\njson schema: {example.model_json_schema()}\n"
        return Prompt(system_prompt=system_prompt, formatted_prompt=input)


task = FunctionGeneratorTask()


def generate(
    num_generations: int = 2,
    batch_size: int = 5,
    checkpoint_strategy=None,
    max_inputs: int = 10,
):
    """Generate functions using GPT-4.
    Args:
        max_input: The maximum number of domains to generate functions for.
    """
    domains = example_function_domain[:max_inputs]
    function_generator = JSONOpenAILLM(
        task=task,
        model="gpt-4-1106-preview",
        max_new_tokens=4096,
    )
    dataset = Dataset.from_dict({"input": domains})
    pipeline = Pipeline(generator=function_generator)
    functions_dataset = pipeline.generate(
        dataset=dataset,
        num_generations=num_generations,
        batch_size=batch_size,
        checkpoint_strategy=checkpoint_strategy,
    )

    functions_dataset = unwrap(functions_dataset)
    return functions_dataset


def unwrap(dataset):
    df = dataset.to_pandas()
    df = df.explode(task.output_args_names)
    dataset = Dataset.from_pandas(df)
    dataset = dataset.rename_column("input", "domain")
    dataset = dataset.rename_column("generations", "function")
    dataset = filter_column_not_none(dataset, "function")
    return dataset
