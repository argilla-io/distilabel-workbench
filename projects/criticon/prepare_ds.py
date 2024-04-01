from datasets import load_dataset
from typing import Dict, Any

dataset_name = "argilla/ultrafeedback-critique"
new_name = "distilabel-internal-testing/ultrafeedback-critique-sft"
local_path = "uf-critique/uf-critique.jsonl"

system_prompt = "You are a critical teacher that provides specific, concise and constructive feedback in plain language, avoid giving me the reference response."

critique_instruction_template = """I need you to give me a score between 1 and 10, where 1 is the worst and 10 is the best, and a critique to show the reason for such a score.

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

score_given_template = """<score>{score}</score>
<critique>{critique}</critique>"""


def prepare_for_sft(example: Dict[str, Any]) -> Dict[str, Any]:
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", type=str, default=dataset_name, help="The name of the base dataset")
    parser.add_argument("--limit", type=int, default=-1, help="Number of rows to generate, defaults to -1 which generates the whole dataset")
    parser.add_argument("--new-name", type=str, default=new_name, help="Name of the new dataset.")
    parser.add_argument("--push-to-hub", action=argparse.BooleanOptionalAction)

    args = parser.parse_args()

    PUSH_TO_HUB = True

    split_name = "train"
    if args.limit > -1:
        split_name += f"[:{args.limit}]"

    print("Loading dataset")
    uf_critique = load_dataset(args.dataset_name, split=split_name)

    column_names = list(uf_critique.column_names)

    uf_critique = uf_critique.map(
        prepare_for_sft,
        num_proc=8,
        remove_columns=column_names,
        desc="Formatting responses with prompt template",
    )

    if args.push_to_hub:
        print("Pushing to hub")
        uf_critique = uf_critique.train_test_split(test_size=0.1, seed=42, shuffle=True)
        uf_critique.push_to_hub(repo_id=args.new_name, private=True)
    else:
        print(f"Saving dataset to file: {local_path}")
        uf_critique.to_json(local_path)

        data_files = {"train": local_path}

        uf_critique = load_dataset("json", data_files=data_files, split="train")
        print("File loaded correctly")
