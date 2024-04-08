"""
pip install mistralai
"""

import os
from typing import Any, Dict, TYPE_CHECKING, List, TypedDict, Optional
import re

from datasets import load_dataset
from distilabel.steps.generators.huggingface import LoadHubDataset
from distilabel.pipeline import Pipeline
from distilabel.llms.mistral import MistralLLM
from distilabel.steps.tasks.text_generation import TextGeneration

if TYPE_CHECKING:
    from distilabel.steps.tasks.typing import ChatType


CRITICON_TEMPLATE = """### Task description:
You are given an instruction, a response to evaluate and the criteria for the feedback and score to take into account.
- You must write the feedback according to the "Feedback criteria", not a general or abstract one.
- After the feedback, write a score as an integer between 1 and 10, using the "Scoring system".
- The output format MUST be: "Feedback: (your feedback) [SCORE] (score between 1 and 10)".

### Feedback criteria:
1. **Correctness & Informativeness**: Does the output provide accurate and helpful information?
2. **Honesty & Uncertainty**: How confidently does the model convey its information, and does it express uncertainty appropriately?
3. **Truthfulness & Hallucination**: Does the model introduce misleading or fabricated details?
4. **Instruction Following**: Does the model's output align with given instructions and the user's intent?

### Scoring system:
1: **Low Quality**: Contains inaccuracies, may be entirely wrong or has severe hallucinations.
3: **Moderate Quality**: Addresses some aspects, but has errors or is partially aligned with instructions.
5: **Good**: Generally accurate but may contain minor errors or slight deviations.
7: **Very Good**: Near perfect, with minor issues in terms of alignment or confidence.
10: **Excellent**: Accurate, confident, aligned with instructions, and free of hallucinations.

### Instruction:
{instruction}

### Response:
{response}

### Feedback:
"""


class CritiqueScore(TypedDict):
    score: Optional[str] = None
    critique: Optional[str] = None
    raw_output: Optional[str] = None


class Criticon(TextGeneration):
    _template: str = CRITICON_TEMPLATE

    @property
    def inputs(self) -> List[str]:
        return ["instruction", "response"]

    @property
    def outputs(self) -> List[str]:
        return ["score", "critique", "raw_output"]

    def format_input(self, input: Dict[str, Any]) -> "ChatType":
        return [{"role": "user", "content": self._template.format(**input)}]

    def format_output(self, output: str | None, input: Dict[str, Any]) -> CritiqueScore:
        score = critique = raw_output = None
        try:
            critique, score = output.split("[SCORE]")
            critique = critique.strip()
            score = score.strip()
        except:
            print(f"Coudln't parse the output: {output}")
            raw_output = output
        
        return CritiqueScore(score=score, critique=critique, raw_output=raw_output)


if __name__ == "__main__":

    with Pipeline(name="benchmark-critique") as pipeline:
        dataset = LoadHubDataset(
            name="load_dataset",
            batch_size=64,
        )
        critique_task = Criticon(
            name="criticon",
            llm=MistralLLM(
                model="mistral-large-latest",
                # model="mistral-medium",
                api_key=os.getenv("MISTRALAI_API_KEY"),  # type: ignore
            ),
            # input_mappings={"instruction": "prompt", "response": "response"},
            input_batch_size=8
        )

        dataset.connect(critique_task)

        distiset = pipeline.run(
            parameters={
                "load_dataset": {
                    "repo_id": "distilabel-internal-testing/mt-bench-eval-critique",
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
            repo_id="distilabel-internal-testing/critique-mt-bench-eval-mistral-large-v0.0",
            private=True,
            token=os.getenv("HF_API_TOKEN"),
        )
