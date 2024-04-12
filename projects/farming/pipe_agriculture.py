import os

from typing import Any, Dict, List

from distilabel.steps.tasks.typing import ChatType

from distilabel.steps.generators.data import LoadDataFromDicts
from distilabel.steps.expand import ExpandColumns
from distilabel.steps.tasks.self_instruct import SelfInstruct
from distilabel.steps.tasks.evol_instruct.base import EvolInstruct
from distilabel.steps.tasks.text_generation import TextGeneration
from distilabel.llms.mistral import MistralLLM
from distilabel.pipeline import Pipeline


# Application description used for SelfInstruct
application_description = """You are an AI assistant than generates queries around the topic of farming and agriculture.
Your should not expect basic but profound questions from your users.
The queries should reflect a diversity of vision and economic positions.
The queries may know about different methods of agriculture and agronomy.
The queries can be positioned politically, economically, or socially.
Also take into account the impact of diverse causes on diverse domains."""


# Topics and positions/perspectives, this is (should) be improved via the UI
topics = ["environmental impact", "agroeconomic efficiency", "land", "animal welfare"]
positions = ["family farming", "agribusiness"]

def create_topics(topics: List[str], positions: List[str]) -> List[str]:
    return [f"{topic} from a {position} perspective" for topic in topics for position in positions]

terms = create_topics(topics, positions)


domain_expert_prompt = """You will be asked about family farming and agribusiness related topics, from different perspectives.
Your answer should be logical and supported by facts, don't fabricate arguments.
Try to gather a diverse point of view taking into account current theories in agronomy, biology, economics, anthropology and ecology.
This is the the instruction:
{instruction}"""


class DomainExpert(TextGeneration):
    """A customized task to generate text as a domain expert in the domain of farming and agriculture.
    """
    _system_prompt: str = "You are a domain expert in the domain of farming and agriculture."
    _template: str = domain_expert_prompt

    def format_input(self, input: Dict[str, Any]) -> "ChatType":
        return [
            {
                "role": "system", "content": self._system_prompt,
                "role": "user", "content": self._template.format(**input)
            }
        ]


with Pipeline("farming") as pipeline:

    load_data = LoadDataFromDicts(
        name="load_data",
        data=[{"input": term} for term in terms],
        batch_size=64,
    )
    base_llm = MistralLLM(
        model="mistral-medium",
        api_key=os.getenv("MISTRAL_API_KEY")
    )
    expert_llm = MistralLLM(
        model="mistral-large-latest",
        api_key=os.getenv("MISTRAL_API_KEY")
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
        input_mappings={"instruction": "questions"}
    )
    expand_instructions = ExpandColumns(name="expand_columns", columns={"instructions": "questions"})
    expand_evolutions = ExpandColumns(name="expand_columns_evolved", columns={"evolved_instructions": "evolved_questions"})

    domain_expert = DomainExpert(
        name="domain_expert",
        llm=expert_llm,
        input_batch_size=8,
        input_mappings={"instruction": "evolved_questions"},
        output_mappings={"generation": "domain_expert_answer"}
    )

    load_data.connect(self_instruct)
    self_instruct.connect(expand_instructions)
    expand_instructions.connect(evol_instruction_complexity)
    evol_instruction_complexity.connect(expand_evolutions)
    expand_evolutions.connect(domain_expert)


if __name__ == "__main__":
    distiset = pipeline.run(
        parameters={
            "domain_expert": {
                "generation_kwargs": {
                    "max_new_tokens": 1024
                },
            },
        }
    )
    distiset.push_to_hub(
        repo_id="distilabel-internal-testing/farming-research-v0.2",
        private=True,
    )
