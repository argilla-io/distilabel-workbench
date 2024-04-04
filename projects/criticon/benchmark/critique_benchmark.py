"""
pip install mistralai
"""

import os
from typing import Any, Dict, TYPE_CHECKING, List, TypedDict
import re

from datasets import load_dataset
# from distilabel.steps.generators.huggingface import LoadHubDataset
from distilabel.steps.generators.data import LoadDataFromDicts
from distilabel.pipeline import Pipeline
from distilabel.llms.mistral import MistralLLM
from distilabel.steps.tasks.text_generation import TextGeneration

if TYPE_CHECKING:
    from distilabel.steps.tasks.typing import ChatType


CRITICON_TEMPLATE = """I need you to give me a score between 1 and 10, where 1 is the worst and 10 is the best, and a critique to show the reason for such a score.

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
        ds = load_dataset("allenai/reward-bench", split="train")

        # Hacky way of just getting a small amount of data for testing        
        data = [
            row for row in ds.filter(lambda x: x["subset"] == "mt-bench-easy").select(range(2))
        ]
        dataset = LoadDataFromDicts(
            name="dataset",
            data=data,
        )
        critique_task = Criticon(
            name="criticon",
            llm=MistralLLM(
                model="mistral-medium",
                api_key=os.getenv("MISTRALAI_API_KEY"),  # type: ignore
            ), # TODO: update to use the correct LLM
            input_mappings={"instruction": "prompt", "response": "chosen"},
        )

        dataset.connect(critique_task)

        distiset = pipeline.run(
            parameters={
                "criticon": {
                    "generation_kwargs": {
                        "max_length": 512, "temperature": 0.7
                    },
                }
            }
        )
        print("EXAMPLE: ")
        import json
        print(json.dumps(distiset["criticon"]["train"].select_columns(["score", "critique"])[:], indent=2))
