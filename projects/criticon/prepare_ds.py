"""
$ python prepare_ds.py \
    --dataset-name "argilla/ultrafeedback-critique" \
    --new-name "distilabel-internal-testing/ultrafeedback-critique-sft-v0.1" \
    --round \
    --push-to-hub

"""

from datasets import load_dataset
from typing import Dict, Any

dataset_name = "argilla/ultrafeedback-critique"
new_name = "distilabel-internal-testing/ultrafeedback-critique-sft-v0.1"
local_path = "uf-critique/uf-critique.jsonl"

system_prompt = "You are a critical teacher that provides specific, concise and constructive feedback in plain language, together with your score."

critique_instruction_template = """### Task description:
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

score_given_template = """{critique} [SCORE] {score}"""


def prepare_for_sft(example: Dict[str, Any], do_round: bool = False) -> Dict[str, Any]:
    if do_round:
        rounding = lambda x: round(float(x))
    else:
        rounding = lambda x: x
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
                score=rounding(example["overall_score"]),
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
    parser.add_argument(
        "--round",
        action=argparse.BooleanOptionalAction,
        help="Whether to round the original scores to work only with integers."
    )
    parser.add_argument(
        "--rescale",
        action=argparse.BooleanOptionalAction,
        help="Whether to rescale the scores to be in the range 1-5, or let them 1-10."
    )
    parser.add_argument("--push-to-hub", action=argparse.BooleanOptionalAction)

    args = parser.parse_args()

    split_name = "train"
    if args.limit > -1:
        split_name += f"[:{args.limit}]"

    print("Loading dataset")
    uf_critique = load_dataset(args.dataset_name, split=split_name)

    if args.round:
        import functools
        prepare_for_sft = functools.partial(prepare_for_sft, do_round=True)
        
    else:
        pass

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
