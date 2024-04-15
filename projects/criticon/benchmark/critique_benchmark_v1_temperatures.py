
import os
from typing import Any, Dict, TYPE_CHECKING, List, TypedDict, Optional

from distilabel.steps.generators.huggingface import LoadHubDataset
from distilabel.pipeline import Pipeline
#from distilabel.llms.openai import OpenAILLM
from distilabel.llms.vllm import vLLM
from distilabel.steps.tasks.text_generation import TextGeneration
import logging

logger = logging.getLogger("criticon-bench")

if TYPE_CHECKING:
    from distilabel.steps.tasks.typing import ChatType


system_prompt = "You are a critical teacher that provides specific, concise and constructive feedback in plain language, together with your score."

CRITICON_TEMPLATE = """### Task description:
You are given an instruction, a response to evaluate and the criteria for the feedback and score to take into account.
- You must write the feedback according to the "Feedback criteria", not a general or abstract one.
- After the feedback, write a score as an integer between 1 and 10, using the "Scoring system".
- The output format MUST be: "(your feedback) [SCORE] (score between 1 and 10)".

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
        return [
            {
                "role": "system",
                "content": system_prompt,
                "role": "user",
                "content": self._template.format(**input)
            }
        ]

    def format_output(self, output: str | None, input: Dict[str, Any]) -> CritiqueScore:
        score = critique = raw_output = None
        try:
            critique, score = output.split("[SCORE]")
            critique = critique.strip()
            score = score.strip()
        except:
            print("***")
            print(f"Couldn't parse the output:\n {output}")
            print("***")
            # logger.info(f"Couldn't parse the output:\n {output}")
            raw_output = output
        
        return CritiqueScore(score=score, critique=critique, raw_output=raw_output)


with Pipeline(name="benchmark-critique") as pipeline:
    dataset = LoadHubDataset(
        name="load_dataset",
        batch_size=64,
    )
    #llm = OpenAILLM(
    #    # model="gpt-3.5-turbo",
    #    model="gpt-4-turbo",
    #    api_key=os.getenv("OPENAI_API_KEY"),
    #)
    llm = vLLM(
        model="distilabel-internal-testing/criticon-sft-v0.1",
    )
    critique_task = Criticon(
        name="criticon",
        llm=llm,
        input_batch_size=64
    )

    dataset.connect(critique_task)


if __name__ == "__main__":

    for temperature in [0.1, 0.5, 0.7, 1]:
        distiset = pipeline.run(
            parameters={
                "load_dataset": {
                    "repo_id": "distilabel-internal-testing/prometheus-bench-critique",
                    "split": "test",
                    "token": os.getenv("HF_API_TOKEN"),
                },
                "criticon": {
                    "llm": {
                        "generation_kwargs": {
                            "max_new_tokens": 1024,
                            "temperature": temperature,
                            "top_p": 0.95
                        },  
                    }
                }
            },
            use_cache=False
        )
        distiset.push_to_hub(
            repo_id=f"distilabel-internal-testing/critique-bench-criticon-sft-temperature{temperature}-v0.1",
            # repo_id="distilabel-internal-testing/critique-bench-gpt-4-turbo-v0.1",
            private=True,
            token=os.getenv("HF_API_TOKEN"),
        )
