import os
import re
from textwrap import dedent
from typing import List, TypedDict

from distilabel.llms.huggingface import InferenceEndpointsLLM
from distilabel.pipeline import Pipeline
from distilabel.steps import LoadHubDataset, PushToHub
from distilabel.steps.tasks import TextGeneration
from dotenv import load_dotenv

load_dotenv()


class Rating(TypedDict):
    """A `TypedDict` representing a rating."""

    value: int
    description: str


class PromptEvaluatorOutput(TypedDict):
    """A `TypedDict` representing the output of an `PromptEvaluationTask`."""

    rating: float
    rationale: str


system_prompt: str = (
    "You are an AI prompt evaluator focused on rating prompts that are clear, interesting and complex for fine-tuning open source LLMs."
)

task_description = dedent(
    """You need to assign a rating to each prompt thinking about the complexity \
for an assistant and if the intent is clear. A very good prompt is one that \
is challenging but also very clear in the intent of the user.

An example of a good prompt involves the following aspects:
- The intent of the user is clear.
- The question, instruction or task for the assistant is challenging or \
    interesting because it involves solving a complex problem, reasoning, involving being creative, etc.

In the case that you feel unequipped of rating a specific prompt, please rate it with -1.

**Scoring**: Rate outputs 1 to 5 based on the following aspects:
"""
)

prompt_evaluator = """{task_description}
{ratings}

This is the prompt:
{input}

Your answer must be in the following format:

<rating>[1-5]</rating>
<rationale>your rationale</rationale>

Please rate the prompt and provide a rationale for your rating."""


ratings = [
    Rating(
        value=1,
        description=dedent(
            """**Very Bad**:\n The prompt doesn't communicate \
            its purpose, is non-sensical or is in a language other than English. \
            The prompt assumes the usage of tools or capabilities that don’t \
            apply to this model, like generating an image or scraping a website."""
        ),
    ),
    Rating(
        value=2,
        description=dedent(
            """**Bad**:\n Suggests a goal but lacks clarity and coherence."""
        ),
    ),
    Rating(
        value=3,
        description="**Ok**:\n The intent is understandable, but it's missing \
            information to complete the task.",
    ),
    Rating(
        value=4,
        description="**Good**:\n Presents a clear goal and necessary information, \
            effectively directing the AI, but the prompt could be more specific.",
    ),
    Rating(
        value=5,
        description="**Very Good**:\n Comprehensive and explicit, leaving no \
            room for ambiguity. Perfectly guides the AI and includes details.",
    ),
]


class PromptEvaluationTask(TextGeneration):
    """Rough translation from the guidelines for the labelling task:
    https://dibt-prompt-collective.hf.space/dataset/f31dabc5-12d5-4845-8361-d41be905d808/settings
    to a distilabel task.
    """

    _system_prompt = system_prompt
    _template = prompt_evaluator

    @property
    def outputs(self) -> List[str]:
        return ["rating", "rationale"]

    def format_input(self, input: dict) -> "Prompt":
        written_ratings = "\n".join(
            [f"{rating['value']}. {rating['description']}" for rating in ratings]
        )
        render_kwargs = {
            "task_description": task_description,
            "ratings": written_ratings,
            "input": input["instruction"],
        }

        return [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": self._template.format(**render_kwargs)},
        ]

    def format_output(self, output: str, input: dict):  # type: ignore
        """Parses the output of the model into the desired format."""

        # Adjusted pattern to handle potentially unclosed <rationale> tag
        pattern = r"<rating>(\d+)</rating>\s*<rationale>(.*?)(?:</rationale>|$)"
        match = re.findall(pattern, output, re.DOTALL)
        if match:
            return {
                "rating": float(match[0][0]),
                "rationale": match[0][1].strip(),
            }
        else:
            return {"rating": -1, "rationale": "No rationale provided."}


with Pipeline(name="prompt_evaluator_pipeline") as pipeline:

    ds = LoadHubDataset(name="load_dataset")

    llm = InferenceEndpointsLLM(
        api_key=os.environ.get("HF_TOKEN"),
        base_url="https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-70B-Instruct",
    )

    prompt_evaluator = PromptEvaluationTask(
        name="prompt_evaluator",
        llm=llm,
        input_batch_size=8,
        input_mappings={"instruction": "prompt"},
    )

    push = PushToHub(
        name="push_to_hub",
    )

ds.connect(prompt_evaluator)
prompt_evaluator.connect(push)

if __name__ == "__main__":
    from argparse import ArgumentParser

    default_load_dataset = "DIBT/10k_prompts_ranked"
    default_push_dataset = "burtenshaw/DIBT_prompts_ranked_synthetic_llama3_70b"
    parser = ArgumentParser()
    parser.add_argument(
        "--load_dataset",
        type=str,
        help="The dataset to load.",
        default=default_load_dataset,
    )
    parser.add_argument(
        "--push_dataset",
        type=str,
        help="The repo to push to.",
        default=default_push_dataset,
    )
    args = parser.parse_args()

    distiset = pipeline.run(
        parameters={
            "load_dataset": {
                "repo_id": args.load_dataset,
                "split": "train",
            },
            "push_to_hub": {
                "repo_id": args.push_dataset,
                "split": "train",
            },
        }
    )
