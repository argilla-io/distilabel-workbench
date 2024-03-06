## Prepare the data

Assumes we already have generated the SPIN dataset, and we will use as "chosen" responses those from `mistral-large` and as "rejected" the ones from the base model we are going to fine-tune, similar to SPIN.

```console
python dpo/data_preparation/prepare_for_training.py \
    --dataset "argilla/10k_prompts_SPIN_iter0_zephyr_top" \
    --target-dataset "argilla/10k_prompts_dpo"
```

## Fine tune using SPIN

### On Runpod

Runpod setup:
- 1 A100 80Gb
- 300Gb container/volume (EACH!)
Use the `train` template image and everything else is setup.

### Connected to the pod

```console
huggingface-cli login --token $HF_API_TOKEN
wandb login $WANDB_TOKEN
```

Move the configuration file and the `finetune.sh` to the root of the repo, and run:

```console
bash finetune.sh
```

> [!NOTE]
> There was a bug in the current version the model was trained, should be fixed in the next release setting `ref_model` to `None` for `PEFT`.

