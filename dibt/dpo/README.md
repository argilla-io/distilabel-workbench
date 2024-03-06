## Prepare the data

Assumes we already have generated the SPIN dataset, and we will use as "chosen" responses those from `mistral-large` and as "rejected" the ones from the base model we are going to fine-tune, similar to SPIN.

```console
python dpo/data_preparation/prepare_for_training.py \
    --dataset "argilla/10k_prompts_SPIN_iter0_zephyr_top" \
    --target-dataset "argilla/10k_prompts_dpo"
```
