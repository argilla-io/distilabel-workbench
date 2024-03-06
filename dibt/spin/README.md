## Prepare the data

### iter0

During the first iteration (iter0), these are the scripts used:

- `generate_reference_spin.py`
    Script to generate the reference responses, uses `mistral-large`.

    Dataset: [argilla/10k_prompts_ranked_with_responses](https://huggingface.co/datasets/argilla/10k_prompts_ranked_with_responses)

- `generate_iter_spin.py`
    Script to generate the initial "generated" responses, from the SFT model that will then be fine-tuned.

    Dataset: [argilla/10k_prompts_ranked_sft_zephyr](https://huggingface.co/datasets/argilla/10k_prompts_ranked_sft_zephyr)

    For zephyr run with the following command:

    ```console
    python generate_iter_spin.py \
        --hf-apikey $HF_API_TOKEN \
        --source-dataset "DIBT/10k_prompts_ranked" \
        --new-dataset "argilla/10k_prompts_ranked_sft_zephyr" \
        --model-name "alignment-handbook/zephyr-7b-sft-full" \
        --batch-size 128 \
        --cuda-devices "0,1"
    ```

- `prepare_for_training.py`
    Generates the dataset that will be directly ingested in the SPINTrainer.

    Dataset: [argilla/10k_prompts_top_SPIN_iter0](https://huggingface.co/datasets/argilla/10k_prompts_top_SPIN_iter0)

    Running the following for zephyr: 

    ```console
    python prepare_for_training.py \
        --portion top \
        --target-dataset argilla/10k_prompts_SPIN_iter0_zephyr_top
    ```

### iter1

- `generate_iter_spin.py`

    Regenerates the "generated" responses from the model in the previous iteration:

    For OpenHermes2.5

    ```console
    python generate_iter_spin.py \
        --hf-apikey $HF_API_TOKEN \
        --source-dataset "argilla/10k_prompts_top_SPIN_iter0" \
        --new-dataset "argilla/10k_prompts_top_SPIN_iter1_generated" \
        --model-name "argilla/OpenHermes-2.5-Mistral-7B-top-SPIN-iter0" \
        --batch-size 128 \
        --cuda-devices "0,1"
    ```

    For zephyr model:

    ```console
    python generate_iter_spin.py \
        --hf-apikey $HF_API_TOKEN \
        --source-dataset "argilla/10k_prompts_SPIN_iter0_zephyr_top" \
        --new-dataset "argilla/10k_prompts_SPIN_iter1_zephyr_top_generated" \
        --model-name "plaguss/zephyr-7b-spin-iter0-v0" \
        --batch-size 128 \
        --cuda-devices "0,1"
    ```

    Dataset: [argilla/10k_prompts_top_SPIN_iter1_generated](https://huggingface.co/datasets/argilla/10k_prompts_top_SPIN_iter1_generated)

- `transform_iter_generated.py`

    For OpenHermes:

    ```console
    python transform_iter_generated.py \
        --real-dataset "argilla/10k_prompts_ranked_with_responses" \
        --generated-dataset "argilla/10k_prompts_top_SPIN_iter1_generated_v2" \
        --new-dataset "argilla/10k_prompts_SPIN_iter1_v2"
    ```

    For zephyr:

    ```console
    python transform_iter_generated.py \
        --real-dataset "argilla/10k_prompts_ranked_with_responses" \
        --generated-dataset "argilla/10k_prompts_SPIN_iter1_zephyr_top_generated" \
        --new-dataset "argilla/10k_prompts_SPIN_iter1_zephyr_top"
    ```


## Fine tune using SPIN

### On Runpod

Runpod setup:
- 4 A100 80Gb
- 500Gb container/volume (EACH!)

### Connected to the pod

And follow the instructions from SPIN repo:

```console
pip install torch==2.1.1 --index-url https://download.pytorch.org/whl/cu121
```

Clone and install the repo from source:

```console
git clone https://github.com/uclaml/SPIN.git && cd SPIN
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
wandb login $WANDB_TOKEN
export WANDB_ENTITY="argilla-io"
export WANDB_PROJECT="dibt-spin-zephyr"
export WANDB_NAME="zephyr-7b-spin-iter0-v0"
```

Overwrite the config files from the original repo with these ones, and add the `finetune.sh` script:

Run the script 

```console
bash scripts/finetune.sh
```

wandb runs:

- OpenHermes 2.5:

    With `avg_rating>=3 & num_responses>1`:

    - [argilla-io/dibt-top-spin-iter0](https://wandb.ai/argilla-io/dibt-top-spin-iter0/runs/ppqznjlm?workspace=user-plaguss-argilla)

    - [argilla-io/dibt-top-spin-iter1](https://wandb.ai/argilla-io/dibt-top-spin-iter1?workspace=user-plaguss-argilla)

- Zephyr

    With `avg_rating>=4 & num_responses>1`:

    - [argilla-io/dibt-top-spin-iter0-zephyr](https://wandb.ai/argilla-io/dibt-spin-zephyr/runs/439olh1m?nw=nwuserplagussargilla)

    - [argilla-io/dibt-top-spin-iter1-zephyr](https://wandb.ai/argilla-io/dibt-spin-zephyr/runs/q938reyu?nw=nwuserplagussargilla)
