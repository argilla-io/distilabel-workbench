# Criticon

> [!NOTE]
> WORK IN PROGRESS

This project contains the work to get `argilla/criticon-v0.1` critique model.

- This [distilabel-internal-testing/ultrafeedback-critique-sft-v0.1](https://huggingface.co/datasets/distilabel-internal-testing/ultrafeedback-critique-sft-v0.1) is the base dataset to train the model.

- [`configs`](./configs/) contains the configuration files for training.

    - `config_full_v2.yaml`: Config file for the full SFT fine tuning.

- [`prepare_ds.py`](./prepare_ds.py) prepares the training script to fine tune a model using SFT.

## Prompt

```python
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
```

## Training script

Connect to Runpod, use the `train` template that comes with the train setup.

Login to HF for the datasets/models and WANDB to track the experiments:

```console
huggingface-cli login --token $HF_API_TOKEN
wandb login $WANDB_TOKEN
```

Run the following script after placing a folder with the config file:

```console
bash finetune_full_v2.sh
```

## WandB Runs

- [SFT-full](https://wandb.ai/argilla-io/criticon/runs/g8sd5jhi?nw=nwuserplagussargilla)

## Notion notes

- [Notion critique model](https://www.notion.so/argilla/Critique-Model-CM-d38933d2200c471f862a2e4bbd6e7a1f)