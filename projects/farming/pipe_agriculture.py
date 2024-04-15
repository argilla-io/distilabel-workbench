import json
import os
from typing import Any, Dict, List, Union, Optional, TypedDict

from distilabel.steps.tasks.typing import ChatType

from distilabel.steps.generators.data import LoadDataFromDicts
from distilabel.steps.expand import ExpandColumns
from distilabel.steps.keep import KeepColumns
from distilabel.steps.tasks.self_instruct import SelfInstruct
from distilabel.steps.tasks.evol_instruct.base import EvolInstruct
from distilabel.steps.tasks.text_generation import TextGeneration
from distilabel.llms.mistral import MistralLLM
from distilabel.llms.openai import OpenAILLM
from distilabel.pipeline import Pipeline
from distilabel.steps import StepInput, StepOutput, Step

from dotenv import load_dotenv

load_dotenv()

# Application description used for SelfInstruct
application_description = """You are an AI assistant than generates queries around the topic of farming and agriculture.
Your should not expect basic but profound questions from your users.
The queries should reflect a diversity of vision and economic positions.
The queries may know about different methods of agriculture and agronomy.
The queries can be positioned politically, economically, or socially.
Also take into account the impact of diverse causes on diverse domains."""


# Topics and positions/perspectives, this is (should) be improved via the UI
# topics = ["environmental impact", "agroeconomic efficiency", "land", "animal welfare"]
# positions = ["family farming", "agribusiness"]

seed_data_path = "domain-specific-seed/farming_defaults.json"

with open(seed_data_path, "r") as f:
    seed_data = json.load(f)

topics = seed_data["topics"]
positions = seed_data["perspectives"]
examples = seed_data["examples"][:5]

examples_prompt = f""" Examples of high quality questions:"""

for example in examples:
    examples_prompt += (
        f"""\n- Question: {example["question"]}\n  Answer: {example["answer"]}\n"""
    )

application_description += examples_prompt


def create_topics(topics: List[str], positions: List[str]) -> List[str]:
    return [
        f"{topic} from a {position} perspective"
        for topic in topics
        for position in positions
    ]


terms = create_topics(topics, positions)


domain_expert_prompt = """You will be asked about family farming and agribusiness related topics, from different perspectives.
Your answer should be logical and supported by facts, don't fabricate arguments.
Try to gather a diverse point of view taking into account current theories in agronomy, biology, economics, anthropology and ecology.
This is the the instruction:
{instruction}"""


class DomainExpert(TextGeneration):
    """A customized task to generate text as a domain expert in the domain of farming and agriculture."""

    _system_prompt: str = (
        "You are a domain expert in the domain of farming and agriculture."
    )
    _template: str = domain_expert_prompt + examples_prompt

    def format_input(self, input: Dict[str, Any]) -> "ChatType":
        return [
            {
                "role": "system",
                "content": self._system_prompt,
                "role": "user",
                "content": self._template.format(**input),
            }
        ]


class CleanNumberedList(Step):
    """A step to clean the numbered list of questions."""

    def process(self, inputs: StepInput) -> StepOutput:
        import re
        pattern = r'^\d+\.\s'

        for input in inputs:
            input["question"] = re.sub(pattern, "", input["question"])
        yield inputs



DOMAIN_EXPERT_CRITIQUE = """### Task description:
You are given an instruction, a response to evaluate and the criteria for the feedback and score to take into account.
- You must write the feedback according to the "Feedback criteria", not a general or abstract one.
- After the feedback, write a score as an integer between 1 and 5, using the "Scoring system".
- The output format MUST be: "(your feedback) [SCORE] (score between 1 and 5)".

### Feedback criteria:
1. Is the answer relevant to the question?
2. Does the answer contain accurate information?
3. Does the answer provide a comprehensive response?
4. Does the answer align with the user's intent?

### Scoring system:
1: **Low Quality**: Contains inaccuracies, may be entirely wrong or has severe hallucinations.
2: **Moderate Quality**: Addresses some aspects, but has errors or is partially aligned with instructions.
3: **Good**: Generally accurate but may contain minor errors or slight deviations.
4: **Very Good**: Near perfect, with minor issues in terms of alignment or confidence.
5: **Excellent**: Accurate, confident, aligned with instructions, and free of hallucinations.

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


class RelevanceLabeller(TextGeneration):
    """A critique model to determine the relevance of an answer in relation to a question."""

    _system_prompt: str = (
        "You are an AI critique in charge of determining the suitability of questions to answers."
    )
    _template: str = DOMAIN_EXPERT_CRITIQUE

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
                "content": self._template.format(**input),
            }
        ]

    def format_output(
        self, output: Union[str, None], input: Dict[str, Any]
    ) -> Dict[str, Any]:
        score = critique = raw_output = None
        try:
            critique, score = output.split("[SCORE]")
            critique = critique.strip()
            score = score.strip()
        except:
            raw_output = output
        
        return CritiqueScore(score=score, critique=critique, raw_output=raw_output)


with Pipeline("farming") as pipeline:

    load_data = LoadDataFromDicts(
        name="load_data",
        data=[{"input": term} for term in terms],
        batch_size=64,
    )
    base_llm = MistralLLM(model="mistral-medium", api_key=os.getenv("MISTRAL_API_KEY"))
    expert_llm = MistralLLM(
        model="mistral-large-latest", api_key=os.getenv("MISTRAL_API_KEY")
    )

    self_instruct = SelfInstruct(
        name="self-instruct",
        application_description=application_description,
        num_instructions=5,
        input_batch_size=8,
        llm=base_llm,
    )

    evol_instruction_complexity = EvolInstruct(
        name="evol_instruction_complexity",
        llm=base_llm,
        num_evolutions=2,
        store_evolutions=True,
        input_batch_size=8,
        include_original_instruction=True,
        input_mappings={"instruction": "question"},
    )
    expand_instructions = ExpandColumns(
        name="expand_columns", columns={"instructions": "question"}
    )
    cleaner = CleanNumberedList(name="clean_numbered_list")
    expand_evolutions = ExpandColumns(
        name="expand_columns_evolved",
        columns={"evolved_instructions": "evolved_questions"},
    )

    domain_expert = DomainExpert(
        name="domain_expert",
        llm=expert_llm,
        input_batch_size=8,
        input_mappings={"instruction": "evolved_questions"},
        output_mappings={"generation": "domain_expert_answer"},
    )
    keep_columns = KeepColumns(name="keep_columns", columns=["model_name", "evolved_questions", "domain_expert_answer"])

    labeller = RelevanceLabeller(
        name="relevance_labeller",
        llm=OpenAILLM(
            model="gpt-4-turbo",
            api_key=os.getenv("OPENAI_API_KEY"),
        ),
        input_batch_size=8,
        input_mappings={"instruction": "evolved_questions", "response": "domain_expert_answer"}
    )

    load_data.connect(self_instruct)
    self_instruct.connect(expand_instructions)
    expand_instructions.connect(cleaner)
    cleaner.connect(evol_instruction_complexity)
    evol_instruction_complexity.connect(expand_evolutions)
    expand_evolutions.connect(domain_expert)
    domain_expert.connect(keep_columns)
    keep_columns.connect(labeller)


if __name__ == "__main__":
    distiset = pipeline.run(
        parameters={
            "relevance_labeller": {
                "llm": {
                    "generation_kwargs": {
                        "max_new_tokens": 1024
                    },
                },
            },
        },
        use_cache=True
    )
    distiset.push_to_hub(
        repo_id="distilabel-internal-testing/farming-research-v0.2",
        private=True,
    )
