# Criticon

> [!NOTE]
> WORK IN PROGRESS

This project contains the work to get `argilla/criticon-v0.1` critique model.

- This [dataset](https://huggingface.co/datasets/argilla/ultrafeedback-critique) is the base dataset to train the model.

- [`configs`](./configs/) contains the configuration files for training.

    - `config_qlora.yaml`: *For testing*, a fine tune config file to check it works as expected.

    - `config_full.yaml`: Config file for the full SFT fine tuning.

- [`prepare_ds.py`](./prepare_ds.py) prepares the training script to fine tune a model using SFT.

## Prompt

```python
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
```

## Training script

Connect to Runpod, use the `train-v1` template that comes with the train setup.

Login to HF for the datasets/models and WANDB to track the experiments:

```console
huggingface-cli login --token $HF_API_TOKEN
wandb login $WANDB_TOKEN
```

Run the following script after placing a folder with the config file:

```console
bash finetune.sh
```
