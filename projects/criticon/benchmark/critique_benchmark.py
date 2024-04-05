"""
pip install mistralai
"""

import os
from typing import Any, Dict, TYPE_CHECKING, List, TypedDict
import re

from datasets import load_dataset
from distilabel.steps.generators.huggingface import LoadHubDataset
from distilabel.pipeline import Pipeline
from distilabel.llms.mistral import MistralLLM
from distilabel.steps.tasks.text_generation import TextGeneration

if TYPE_CHECKING:
    from distilabel.steps.tasks.typing import ChatType


CRITICON_TEMPLATE = """I need you to give me a score between 1 and 10, where 1 is the worst and 10 is the best, and a critique to show the reason for such a score.

**Scoring**: These values represent the overall quality, consider all aspects:
1. **Low Quality**: Contains inaccuracies, may be entirely wrong or has severe hallucinations.
3. **Moderate Quality**: Addresses some aspects, but has errors or is partially aligned with instructions.
5. **Good**: Generally accurate but may contain minor errors or slight deviations.
7. **Very Good**: Near perfect, with minor issues in terms of alignment or confidence.
10. **Excellent**: Accurate, confident, aligned with instructions, and free of hallucinations.

These are the criteria to take into account:
1. **Correctness & Informativeness**: Does the output provide accurate and helpful information?
2. **Honesty & Uncertainty**: How confidently does the model convey its information, and does it express uncertainty appropriately?
3. **Truthfulness & Hallucination**: Does the model introduce misleading or fabricated details?
4. **Instruction Following**: Does the model's output align with given instructions and the user's intent?

Consider how good the response follows the instruction:

<instruction>{instruction}</instruction>
<response>{response}</response>

Your answer must be in the following format:

<score>[1-10]</score>
<critique>your critique</critique>"""


class CritiqueScore(TypedDict):
    score: float
    critique: str


class Criticon(TextGeneration):
    _template: str = CRITICON_TEMPLATE
    _regex_pattern = re.compile(r"<[^>]*>([^<]+)<\/[^>]*>", re.IGNORECASE)

    @property
    def inputs(self) -> List[str]:
        return ["instruction", "response"]

    @property
    def outputs(self) -> List[str]:
        return ["score", "critique"]

    def format_input(self, input: Dict[str, Any]) -> "ChatType":
        return [{"role": "user", "content": self._template.format(**input)}]

    def format_output(self, output: str | None, input: Dict[str, Any]) -> CritiqueScore:
        matches = re.findall(self._regex_pattern, output)
        if matches:
            return CritiqueScore(score=float(matches[0]), critique=matches[1])
        return CritiqueScore(score=None, critique=None)
        


if __name__ == "__main__":

    with Pipeline(name="benchmark-critique") as pipeline:
        dataset = LoadHubDataset(
            name="load_dataset",
            batch_size=8,
        )
        critique_task = Criticon(
            name="criticon",
            llm=MistralLLM(
                model="mistral-medium",
                api_key=os.getenv("MISTRALAI_API_KEY"),  # type: ignore
            ),
            input_mappings={"instruction": "prompt", "response": "response"},
            input_batch_size=8
        )

        dataset.connect(critique_task)

        distiset = pipeline.run(
            parameters={
                "load_dataset": {
                    "repo_id": "distilabel-internal-testing/reward-bench-critique-alpacaeval-easy",
                    "split": "train",
                    "token": os.getenv("HF_API_TOKEN"),
                },
                "criticon": {
                    "generation_kwargs": {
                        "max_length": 512, "temperature": 1.0, "top_p": 0.95
                    },
                }
            }
        )
        distiset.push_to_hub(
            repo_id="distilabel-internal-testing/reward-bench-critique-alpacaeval-easy-labeled-v0.2",
            private=True,
            token=os.getenv("HF_API_TOKEN"),
        )
