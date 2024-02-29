"""
$ python dibt/synthetic_evaluator.py
"""

import os
from dataclasses import dataclass
from typing import List, TypedDict

import regex as re
from distilabel import Prompt, TextGenerationTask

from .criterion import Rating, criterion


class PromptEvaluatorOutput(TypedDict):
    """A `TypedDict` representing the output of an `PromptEvaluationTask`."""

    rating: float
    rationale: str


prompt_evaluator = """{task_description}

Instruction:
{instruction}

Response:
{response}

{criterion_name}
Options:
{criterion_options}

Your answer must be in the following format:

<rating>[1-5]</rating>
<rationale>your rationale</rationale>

Please rate the Response: based on the Options: and provide a rationale for your rating."""


@dataclass
class HelmInstructTask(TextGenerationTask):
    """Rough translation from the guidelines for the labelling task:
    https://crfm.stanford.edu/2024/02/18/helm-instruct.html
    to a distilabel task.
    """

    criterion_question: str
    criterion_options: List[Rating]
    task_description: str = (
        "The following is an instruction written by a human, and a response to the instruction written by an AI model. Please answer the following questions about the AI model’s response."
    )
    system_prompt: str = (
        "You are an AI response evaluator focused on rating prompts that are clear, interesting and complex for fine-tuning open source LLMs."
    )

    def generate_prompt(self, prompt: str, response: str) -> Prompt:
        render_kwargs = {
            "task_description": self.task_description,
            "criterion_question": self.criterion_question,
            "criterion_options": self.criterion_options,
            "prompt": prompt,
            "response": response,
        }
        return Prompt(
            system_prompt=self.system_prompt,
            formatted_prompt=prompt_evaluator.format(**render_kwargs),
        )

    @classmethod
    def for_helpfullness(cls):
        return cls(criterion="Helpfulness")

    @classmethod
    def for_understandability(cls):
        return cls(criterion="Understandability")

    @classmethod
    def for_completeness(cls):
        return cls(criterion="Completeness")

    @classmethod
    def for_conciseness(cls):
        return cls(criterion="Conciseness")

    @classmethod
    def for_harmlessness(cls):
        return cls(criterion="Harmlessness")

    def __post_init__(self):
        self.criterion_question = criterion[self.criterion]["question"]
        self.criterion_options = criterion[self.criterion]["ratings"]
        self.task_description = criterion[self.criterion]["question"]

    def parse_output(self, output: str) -> PromptEvaluatorOutput:  # type: ignore
        """Parses the output of the model into the desired format."""
        pattern = r"<rating>(.*?)</rating>\s*<rationale>(.*?)</rationale>"
        match = re.findall(pattern, output, re.DOTALL)
        if match:
            return PromptEvaluatorOutput(
                rating=float(match[0][0]),
                rationale=match[0][1].strip(),
            )


if __name__ == "__main__":
    from datasets import load_dataset
    from distilabel.dataset import DatasetCheckpoint
    from distilabel.llm import OpenAILLM
    from distilabel.pipeline import Pipeline

    dataset = load_dataset("DIBT/10k_prompts_ranked", split="train").rename_column(
        "prompt", "input"
    )

    OPENAI_API_TOKEN = os.getenv("OPENAI_API_TOKEN")
    HF_API_TOKEN = os.getenv("HF_API_TOKEN")
    NEW_DATASET_NAME = "argilla/10k_prompts_ranked_synthetic"

    checkpoint_strategy = DatasetCheckpoint(
        strategy="hf-hub",
        extra_kwargs={
            "repo_id": NEW_DATASET_NAME,
            "token": HF_API_TOKEN,
            "private": True,
            "split": "train",
        },
        save_frequency=500,
    )

    pipe = Pipeline(
        generator=OpenAILLM(
            model="gpt-4-1106-preview",  # gpt-4 turbo
            task=PromptEvaluationTask.for_overall_quality(),
            max_new_tokens=512,
            num_threads=8,
            api_key=OPENAI_API_TOKEN,
            temperature=0.3,
        )
    )
    new_ds = pipe.generate(
        dataset,
        num_generations=1,
        batch_size=16,
        checkpoint_strategy=checkpoint_strategy,
    )
