"""
pip install mistralai
"""

import os
from typing import Any, Dict, TYPE_CHECKING, List, TypedDict, Optional
import re

from distilabel.steps.generators.huggingface import LoadHubDataset
from distilabel.pipeline import Pipeline
from distilabel.llms.huggingface.inference_endpoints import InferenceEndpointsLLM
from distilabel.steps.tasks.text_generation import TextGeneration

if TYPE_CHECKING:
    from distilabel.steps.tasks.typing import ChatType


system_prompt = "You are a critical teacher that provides specific, concise and constructive feedback in plain language, avoid giving me the reference response."

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
    score: Optional[float] = None
    critique: Optional[str] = None
    raw_output: Optional[str] = None


class Criticon(TextGeneration):
    _system_prompt: str = system_prompt
    _template: str = CRITICON_TEMPLATE
    _regex_pattern = re.compile(r"<[^>]*>([^<]+)<\/[^>]*>", re.IGNORECASE)

    @property
    def inputs(self) -> List[str]:
        return ["instruction", "response"]

    @property
    def outputs(self) -> List[str]:
        return ["score", "critique", "raw_output"]

    def format_input(self, input: Dict[str, Any]) -> "ChatType":
        return [
            {
                "role": "system",
                "content": self._system_prompt,
                "role": "user",
                "content": self._template.format(**input)
            }
        ]

    def format_output(self, output: str | None, input: Dict[str, Any]) -> CritiqueScore:
        matches = re.findall(self._regex_pattern, output)
        score = critique = None
        if matches:
            try:
                score = float(matches[0])
                critique = matches[1]
            except:
                print(f"Error parsing output: {output}")

        return CritiqueScore(score=score, critique=critique, raw_output=output)



if __name__ == "__main__":

    with Pipeline(name="benchmark-critique") as pipeline:
        dataset = LoadHubDataset(
            name="load_dataset",
            batch_size=64,
        )
        critique_task = Criticon(
            name="criticon",
            llm=InferenceEndpointsLLM(
                base_url="https://reafhm628q224kuz.us-east-1.aws.endpoints.huggingface.cloud",
                api_key=os.getenv("HF_API_TOKEN"),  # type: ignore
            ),
            input_mappings={"instruction": "prompt", "response": "response"},
            input_batch_size=8
        )

        dataset.connect(critique_task)

        distiset = pipeline.run(
            use_cache=False,
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
            repo_id="distilabel-internal-testing/reward-bench-critique-alpacaeval-easy-labeled-criticon-v0.1",
            private=True,
            token=os.getenv("HF_API_TOKEN"),
        )
