from dataclasses import field
from textwrap import dedent
from typing import ClassVar, Any

from pandas import DataFrame

from distilabel.dataset import Dataset
from distilabel.llm import OpenAILLM
from distilabel.pipeline import Pipeline
from distilabel.tasks import Prompt
from distilabel.tasks.preference.ultrafeedback import Rating, UltraFeedbackTask
from dotenv import load_dotenv

load_dotenv("../.env")

template_path = "/home/ben/code/distilabel-workbench/scripts/function_calling_dataset/templates/functionfeedback.jinja2"


class FunctionFeedbackTask(UltraFeedbackTask):

    __jinja2_template__: ClassVar[str] = template_path

    @property
    def input_args_names(self):
        return ["function", "instructions", "generations"]

    def generate_prompt(
        self, function: str, instructions: str, generations: list[str], **_: Any
    ) -> Prompt:
        """Generates a prompts."""
        render_kwargs = {
            "task_description": self.task_description,
            "ratings": self.ratings,
            "instructions": instructions,
            "responses": generations,
            "function": function,
        }
        return Prompt(
            system_prompt=self.system_prompt,
            formatted_prompt=self.template.render(**render_kwargs),
        )


def generate(
    dataset: "Dataset",
    batch_size: int = 5,
    num_generations: int = 2,
    checkpoint_strategy=None,
    max_inputs: int = None,
    max_row_inputs: int = 2,
) -> "CustomDataset":
    task = FunctionFeedbackTask(
        system_prompt="Your role is to evaluate function calling ability based on given criteria",
        task_description=dedent(
            """
            # Personal Assistant Function Calling Feedback
            Evaluate the model's function calling based on various criteria:
            1. **Correctness**: Does the output provide accurate a relevant function call based on the schema?
            2. **Instruction Following**: Does the function follow the instruction?
            3. **Completeness**: Does the function call supply all relevant parameters?
            4. **Clarity**: Is the function call clear and easy to understand?
            5. **Relevance**: If the function is not relevant to the instruction, the function call should be a null value. 

            **Scoring**: Rate outputs 1 to 3 based on the overall quality, considering all aspects:
            """
        ),
        ratings=[
            Rating(
                value=1,
                description="The function call is incomplete and does not represent the instruction.",
            ),
            Rating(
                value=2,
                description="The function call is complete but field, descriptions, and examples should be improved.",
            ),
            Rating(
                value=3,
                description="The function call is complete and represents the instruction fully.",
            ),
        ],
    )
    labeller = OpenAILLM(
        task=task,
        model="gpt-4",
        max_new_tokens=4096,
    )
    pipeline = Pipeline(labeller=labeller)
    dataset = Dataset.from_list(dataset.to_list()[:max_inputs])
    # dataset = limit_row_input(dataset, max_row_inputs)
    feedback_dataset = pipeline.generate(
        dataset=dataset,
        num_generations=num_generations,
        batch_size=batch_size,
        checkpoint_strategy=checkpoint_strategy,
    )
    return feedback_dataset


def drop_columns(dataset: "Dataset") -> "Dataset":
    dataset = dataset.rename_column("rating", "_rating")
    dataset = dataset.rename_column("feedback", "_feedback")
    return dataset


def limit_row_input(dataset: Dataset, max_inputs: int = 2) -> Dataset:
    df = dataset.to_pandas()
    rows = []
    for idx in df.index:
        row = df.loc[idx]
        _limited_rows = []
        for generations in row.generations:
            _limited_generations = []
            _limited_instructions = []
            for generation, instruction in zip(generations, row.instructions):
                _limited_generations.append(generation)
                _limited_instructions.append(instruction)
                if (
                    len(_limited_generations) >= max_inputs
                    or len(_limited_instructions) >= max_inputs
                ):
                    row = row.copy()
                    row["instructions"] = _limited_instructions
                    row["generations"] = [_limited_generations]
                    _limited_rows.append(row)
                    _limited_generations = []
                    _limited_instructions = []
        rows.extend(_limited_rows)
    limited_df = DataFrame(rows)
    dataset = Dataset.from_pandas(limited_df)
    return dataset
