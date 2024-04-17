
import os
from typing import Any, Dict, TYPE_CHECKING, List, TypedDict, Optional

from distilabel.steps.generators.huggingface import LoadHubDataset
from distilabel.pipeline import Pipeline
from distilabel.llms.vllm import vLLM
from distilabel.steps.tasks.text_generation import TextGeneration
import logging

logger = logging.getLogger("criticon-bench")

if TYPE_CHECKING:
    from distilabel.steps.tasks.typing import ChatType



PROMETHEUS_TEMPLATE = """###Task Description:
An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing a evaluation criteria are given.
1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output format should look as follows: \"Feedback: (write a feedback for criteria) [RESULT] (an integer number between 1 and 5)\"
4. Please do not generate any other opening, closing, and explanations.

###The instruction to evaluate:
{instruction}

###Response to evaluate:
{response}

###Reference Answer (Score 5):


###Score Rubrics:
[Is the model able to critique the overall quality of the response based on the given score rubric?]
Score 1: Contains inaccuracies, may be entirely wrong or has severe hallucinations.
Score 2: Addresses some aspects, but has errors or is partially aligned with instructions.
Score 3: Generally accurate but may contain minor errors or slight deviations.
Score 4: Near perfect, with minor issues in terms of alignment or confidence.
Score 5: Accurate, confident, aligned with instructions, and free of hallucinations.

###Feedback:
"""


class CritiqueScore(TypedDict):
    score: Optional[str] = None
    critique: Optional[str] = None
    raw_output: Optional[str] = None


class Prometheus(TextGeneration):
    _template: str = PROMETHEUS_TEMPLATE

    @property
    def inputs(self) -> List[str]:
        return ["instruction", "response"]

    @property
    def outputs(self) -> List[str]:
        return ["score", "critique", "raw_output"]

    def format_input(self, input: Dict[str, Any]) -> "ChatType":
        return [
            {
                "role": "user",
                "content": self._template.format(**input)
            }
        ]

    def format_output(self, output: str | None, input: Dict[str, Any]) -> CritiqueScore:
        score = critique = raw_output = None
        try:
            critique, score = output.split("[RESULT]")
            critique = critique.strip()
            score = score.strip()
        except:
            raw_output = output
        
        return CritiqueScore(score=score, critique=critique, raw_output=raw_output)


if __name__ == "__main__":

    for temperature in [0.1, 0.3, 0.5, 0.7, 1]:
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
                model="kaist-ai/prometheus-7b-v1.0",
            )
            critique_task = Prometheus(
                name="prometheus-critique",
                llm=llm,
                input_batch_size=64
            )
        
            dataset.connect(critique_task)
        distiset = pipeline.run(
            parameters={
                "load_dataset": {
                    "repo_id": "distilabel-internal-testing/prometheus-bench-critique",
                    "split": "test",
                    "token": os.getenv("HF_API_TOKEN"),
                },
                "prometheus-critique": {
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
            repo_id=f"distilabel-internal-testing/critique-bench-prometheus-temperature{temperature}-v0.1",
            # repo_id="distilabel-internal-testing/critique-bench-gpt-4-turbo-v0.1",
            private=True,
            token=os.getenv("HF_API_TOKEN"),
        )
