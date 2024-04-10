import json
from ast import literal_eval
from itertools import combinations
from typing import Any, Dict

from datasets import Dataset
from distilabel.llm import JSONOpenAILLM
from distilabel.pipeline import Pipeline
from distilabel.tasks import Prompt, TextGenerationTask
from dotenv import load_dotenv
from tqdm import tqdm

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


### DISTRACTORS ###


def distract(
    dataset: Dataset,
    max_distractors: int = 2,
    max_inputs: int = 10,
):
    """Retrieve function distractions from the dataset.
    Args:
        dataset: The dataset of function calls
        num_distractions: The number of distractions to generate for each function call.
        max_input: The maximum number of domains to generate functions for.
    """
    # make a lookup dataframe of function domains to functions
    df = dataset.to_pandas().sample(frac=1)[:max_inputs]
    df = df.drop_duplicates(subset=["function"])
    domain_function_lookup = df.groupby("domain").function.apply(list).to_dict()

    # iterate through the dataset and retieve the function for each domain
    for idx, row in tqdm(df.iterrows()):
        try:
            function = json.loads(row["function"])
            function_name = function["function"]["name"]
            available_functions = []
            for func in domain_function_lookup[row.domain]:
                try:
                    function = json.loads(func)
                    if function["function"]["name"] != function_name:
                        available_functions.append(func)
                except json.JSONDecodeError:
                    continue
        except Exception:  # (json.JSONDecodeError, KeyError)
            continue
        # TODO: we should get the least similar functions to the function for the domain
        # append the distractions to a distractors collumn in the dataset
        for num_distractors in range(1, max_distractors + 1):
            for distractors in combinations(available_functions, num_distractors):
                df.at[idx, "distractors"] = json.dumps(list(distractors))
                df.at[idx, "num_distractors"] = num_distractors
    df = df.loc[:, ~df.columns.str.contains("_index_")]
    dataset = Dataset.from_pandas(df)
    return dataset
