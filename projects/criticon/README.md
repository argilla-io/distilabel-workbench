# Criticon

> [!NOTE]
> WORK IN PROGRESS

This project contains the work to get `argilla/criticon-v0.1` critique model.

- This [dataset](https://huggingface.co/datasets/argilla/ultrafeedback-critique) is the base dataset to train the model.

- [`configs`](./configs/) contains the configuration files for training.

- [`prepare_ds.py`](./prepare_ds.py) prepares the training script to fine tune a model using SFT (*currently hardcoded to generate a small subsample of the original dataset prepared for training an SFT model*).

## Prompt

```python
system_prompt = "User: A one-turn chat between a curious user and an artificial intelligence critique assistant."

critique_instruction_template = """You are a critical teacher that provides specific, concise and constructive feedback for me in plain language, avoid giving me the reference response.

Consider how good the response follows the instruction:

<instruction>{instruction}</instruction>
<response>{response}</response>

Your answer must be in the following format:

<score>[1-10]</score>
<critique>your critique</critique>"""
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
