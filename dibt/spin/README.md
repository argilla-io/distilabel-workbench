## Prepare the data

Scripts:

- `generate_reference_spin.py`
    Script to generate the reference responses, uses `mistral-large`.

    Dataset: [argilla/10k_prompts_ranked_with_responses](https://huggingface.co/datasets/argilla/10k_prompts_ranked_with_responses)

- generate_iter_spin.py
    Script to generate the initial "generated" responses, from the SFT model that will then be fine-tuned.

    Dataset: [argilla/10k_prompts_ranked_sft](https://huggingface.co/datasets/argilla/10k_prompts_ranked_sft)

- `prepare_for_training.py`
    Generates the dataset that will be directly ingested in the SPINTrainer.

    Dataset: [argilla/10k_prompts_top_SPIN_iter0](https://huggingface.co/datasets/argilla/10k_prompts_top_SPIN_iter0)

    Running the following: 

    ```console
    python spin/prepare_for_training.py --portion top --target-dataset argilla/10k_prompts_top_SPIN_iter0
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

### NOTES

- The `batch_size` seemed big, some warnings appeared during training.

- More frequency of logging for the test set, or nothing appears.

- No automatic push to the hub is configured, has to be done manually afterwards:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("/outputs")
tokenizer = AutoTokenizer.from_pretrained("/outputs")
model.push_to_hub("argilla/OpenHermes-2.5-Mistral-7B-top-SPIN-iter0", private=True)
tokenizer.push_to_hub("argilla/OpenHermes-2.5-Mistral-7B-top-SPIN-iter0", private=True)
```

- The model generates weird content. Possible reasons:
    - Dataset to small for a full Fine Tune. -> Decrease to 1 epoch.
    - Bad quality of the dataset. -> Try with the full dataset.
