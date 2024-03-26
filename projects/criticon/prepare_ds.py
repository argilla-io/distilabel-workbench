from datasets import load_dataset
from typing import Dict, Any
from transformers import PreTrainedTokenizer, AutoTokenizer

dataset_name = "argilla/ultrafeedback-critique"
#model_name = "mistralai/Mistral-7B-v0.1"
# The model we want the chat template from
model_name = "teknium/OpenHermes-2.5-Mistral-7B"
local_path = "uf-critique/uf-critique.jsonl"

PUSH_TO_HUB = True

print("Loading dataset")
uf_critique = load_dataset(dataset_name, split="train[:100]")

print("Applying prompt template")

system_prompt = "User: A one-turn chat between a curious user and an artificial intelligence critique assistant."

critique_instruction_template = """You are a critical teacher that provides specific, concise and constructive feedback for me in plain language, avoid giving me the reference response.

Consider how good the response follows the instruction:

<instruction>{instruction}</instruction>
<response>{response}</response>

Your answer must be in the following format:

<score>[1-10]</score>
<critique>your critique</critique>"""

score_given_template = """<score>{score}</score>
<critique>{critique}</critique>"""


def prepare_for_sft(
    example: Dict[str, Any]
    #, tokenizer: PreTrainedTokenizer
) -> Dict[str, Any]:
    # example["messages"] = tokenizer.apply_chat_template(
    example["messages"] = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": critique_instruction_template.format(
                instruction=example["instruction"],
                response=example["response"]
            ),
        },
        {
            "role": "assistant",
            "content": score_given_template.format(
                score=example["overall_score"],
                critique=example["critique"]
            )
        },
    ]
    return example

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.chat_template
column_names = list(uf_critique.column_names)

uf_critique = uf_critique.map(
    prepare_for_sft,
    # fn_kwargs={"tokenizer": tokenizer},
    num_proc=4,
    remove_columns=column_names,
    desc="Formatting responses with prompt template",
)

print(f"Saving to file dataset: {local_path}")
uf_critique.to_json(local_path)

data_files = {"train": local_path}

uf_critique = load_dataset("json", data_files=data_files, split="train")
print("File loaded correctly")

if PUSH_TO_HUB:
    print("Pushing to hub")
    uf_critique = uf_critique.train_test_split(test_size=0.1, seed=42, shuffle=True)
    uf_critique.push_to_hub(repo_id="plaguss/uf-critique-test", private=True)
