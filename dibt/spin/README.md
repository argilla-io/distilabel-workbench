## Prepare the data

WIP

## Fine tune using SPIN

Runpod setup:
- 4 A100 80Gb
- 500Gb container/volume (EACH!)

And follow the instructions from SPIN repo:

```console
pip install torch==2.1.1 --index-url https://download.pytorch.org/whl/cu121
```

Clone and install the repo from source:

```console
git clone https://github.com/uclaml/SPIN.git
cd SPIN
```

Install package and flash-attn

```console
python -m pip install .
python -m pip install flash-attn==2.5.3 --no-build-isolation
```

Log to huggingface:

```console
huggingface-cli login --token $HF_API_TOKEN
```

Log to wandb:

```console
pip install wandb
wandb login
export WANDB_ENTITY="argilla-io"
export WANDB_PROJECT="dibt-top-spin-iter0"
```

Overwrite the config files from the original repo with these ones, and add the `finetune-mine.sh` script:

Run the script 

```console
bash scripts/finetune-my.sh
```

wandb runs:

- [argilla-io/dibt-top-spin-iter0](https://wandb.ai/argilla-io/dibt-top-spin-iter0?workspace=user-plaguss-argilla)
