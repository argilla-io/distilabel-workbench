from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from distilabel.llms import InferenceEndpointsLLM
from distilabel.mixins.runtime_parameters import RuntimeParameter
from distilabel.pipeline import Pipeline
from distilabel.steps import LoadHubDataset
from distilabel.steps.tasks import Task
from jinja2 import Template
from pydantic import Field, PrivateAttr

if TYPE_CHECKING:
    from distilabel.steps.tasks.typing import ChatType

SYSTEM_PROMPT = """
Your task is to generate a conversation of multiple turns between a `User` and an `Assistant`. You will be provided with a document as a `Context`, and you will have to generate a conversation with {{ num_user_messages }} `User` messages and {{ num_assistant_messages }} `Assistant` messages.{% if end_with_user -%} You MUST end the conversation with a `User` message.{%- endif %}


A turn is one user message and one assistant message. Each new turn will continue developing the conversation from the previous turns.

The conversation MUST be engaging.

```markdown
User: <user_interaction_0>
Assistant: <assistant_interaction_0>
User: <user_interaction_1>
Assistant: <assistant_interaction_1>
...
```

You MUST only output the conversation, no additional text.
""".strip()

PROMPT_TEMPLATE = """
## Context

{context}
""".strip()


class GenerateConvWithContext(Task):
    turns: RuntimeParameter[int] = Field(
        default=5,
        description="The number of conversation turns to be generated. A turn is one user"
        " message and one assistant message.",
    )
    end_with_user: RuntimeParameter[bool] = Field(
        default=False,
        description="Whether the conversation should end with a user message.",
    )

    _template: Union[Template, None] = PrivateAttr(...)

    def load(self) -> None:
        super().load()

        self._template = Template(SYSTEM_PROMPT)

    @property
    def inputs(self) -> List[str]:
        return ["context"]

    def format_input(self, input: Dict[str, Any]) -> "ChatType":
        assert self._template, "Template not loaded."

        return [
            {
                "role": "system",
                "content": self._template.render(
                    num_user_messages=self.turns + 1  # type: ignore
                    if self.end_with_user
                    else self.turns,
                    num_assistant_messages=self.turns,
                    end_with_user=self.end_with_user,
                ),
            },
            {
                "role": "user",
                "content": PROMPT_TEMPLATE.format(**{**input, "turns": self.turns}),
            },
        ]

    @property
    def outputs(self) -> List[str]:
        return ["conversation"]

    def format_output(
        self, output: Union[str, None], input: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        if output is None:
            return {"conversation": None}

        lines = output.strip().split("\n")

        conversation = []
        for line in lines:
            role, content = line.split(":", 1)
            conversation.append({"role": role, "content": content.strip()})

        return {"conversation": conversation}


with Pipeline(name="fine-preferences") as pipeline:
    load_dataset = LoadHubDataset(
        repo_id="distilabel-internal-testing/fineweb-edu-subset",
        split="train",
        output_mappings={"text": "context"},
    )

    generate_conversation = GenerateConvWithContext(
        turns=5,
        end_with_user=True,
        llm=InferenceEndpointsLLM(
            model_id="meta-llama/Meta-Llama-3-70B-Instruct",
            tokenizer_id="meta-llama/Meta-Llama-3-70B-Instruct",
        ),
    )

    load_dataset >> generate_conversation


if __name__ == "__main__":
    distiset = pipeline.run(
        parameters={
            generate_conversation.name: {
                "llm": {
                    "generation_kwargs": {"temperature": 0.7, "max_new_tokens": 4096}
                }
            }
        }
    )

    distiset.push_to_hub("distilabel-internal-testing/fine-preferences-test-2")
