# Running SPIN on DIBT 10K ranked

These README contains the instructions to run [SPIN](https://github.com/uclaml/SPIN) on a subset of the [DIBT/10k_prompts_ranked](https://huggingface.co/datasets/DIBT/10k_prompts_ranked) dataset: Those that have `avg_rating>=4` and `num_response>1`, making a total of 1832 records (which will then be splitted in 1648 for training and 184 for testing).

It contains the references to all the scripts to generate the datasets, the configuration files used for the training process and the setup used to run the model. The dataset generation was done using [distilabel==0.6.0](https://github.com/argilla-io/distilabel).

SPIN needs a specific format for the data to do the training, where the "real" data is the reference for the model to improve. As the dataset was made of prompts, we decided to generate these responses using [`mistral-large`](https://docs.mistral.ai/platform/endpoints/). The different iterations of the "generated" datasets were created using `distilabel` with `vllm`, using 2 A100 GPUs (just for speed, it should work with less computer power, just need to update the `--cuda-devices` and `--batch-size` arguments accordingly).

## Prepare the data

Initially, we create the reference dataset with the *real* responses being generated from `mistral-large`, using the following script:

- `generate_reference_spin.py`
    Script to generate the reference responses, uses `mistral-large`:

    Dataset: [argilla/10k_prompts_ranked_with_responses](https://huggingface.co/datasets/argilla/10k_prompts_ranked_with_responses)

### Experiment *top* prompts

The following are the steps to prepare the training data for SPIN, and the resulting datasets:

<details><summary> SPIN iter 0 </summary><hr>

- `generate_iter_spin.py`
    Script to generate the initial "generated" responses, from the SFT model that will then be fine-tuned.

    Dataset: [argilla/10k_prompts_ranked_sft_zephyr](https://huggingface.co/datasets/argilla/10k_prompts_ranked_sft_zephyr)

    Run the following:

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
    Generates the dataset that will be directly ingested in the `SPINTrainer`.

    Dataset: [argilla/10k_prompts_top_SPIN_iter0](https://huggingface.co/datasets/argilla/10k_prompts_top_SPIN_iter0)

    Running the following python script: 

    ```console
    python prepare_for_training.py \
        --portion top \
        --target-dataset argilla/10k_prompts_SPIN_iter0_zephyr_top
    ```

</details>


<details><summary> SPIN iter 1 </summary><hr>


- `generate_iter_spin.py`

    Regenerates the "generated" responses from the model in the previous iteration:

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

    The script transforms the generated responses to the format expected by SPIN trainer:

    ```console
    python transform_iter_generated.py \
        --real-dataset "argilla/10k_prompts_ranked_with_responses" \
        --generated-dataset "argilla/10k_prompts_SPIN_iter1_zephyr_top_generated" \
        --new-dataset "argilla/10k_prompts_SPIN_iter1_zephyr_top"
    ```

</details>


<details><summary> SPIN iter 2 </summary><hr>


- `generate_iter_spin.py`

    Regenerates the "generated" responses from the model in the previous iteration:

    ```console
    python generate_iter_spin.py \
        --hf-apikey $HF_API_TOKEN \
        --source-dataset "argilla/10k_prompts_SPIN_iter0_zephyr_top" \
        --new-dataset "argilla/10k_prompts_SPIN_iter2_zephyr_top_generated" \
        --model-name "plaguss/zephyr-7b-spin-iter1-v0" \
        --batch-size 128 \
        --cuda-devices "0,1"
    ```

    Dataset: [argilla/10k_prompts_top_SPIN_iter2_generated](https://huggingface.co/datasets/argilla/10k_prompts_top_SPIN_iter2_generated)

- `transform_iter_generated.py`

    The script transforms the generated responses to the format expected by SPIN trainer:

    ```console
    python transform_iter_generated.py \
        --real-dataset "argilla/10k_prompts_ranked_with_responses" \
        --generated-dataset "argilla/10k_prompts_SPIN_iter2_zephyr_top_generated" \
        --new-dataset "argilla/10k_prompts_SPIN_iter2_zephyr_top"
    ```

</details>

<details><summary> SPIN iter 3 </summary><hr>

- `generate_iter_spin.py`

    Regenerates the "generated" responses from the model in the previous iteration:

    ```console
    python generate_iter_spin.py \
        --hf-apikey $HF_API_TOKEN \
        --source-dataset "argilla/10k_prompts_SPIN_iter0_zephyr_top" \
        --new-dataset "argilla/10k_prompts_SPIN_iter3_zephyr_top_generated" \
        --model-name "plaguss/zephyr-7b-spin-iter2-v0" \
        --batch-size 128 \
        --cuda-devices "0,1"
    ```

    Dataset: [argilla/10k_prompts_top_SPIN_iter3_generated](https://huggingface.co/datasets/argilla/10k_prompts_top_SPIN_iter3_generated)

- `transform_iter_generated.py`

    The script transforms the generated responses to the format expected by SPIN trainer:

    ```console
    python transform_iter_generated.py \
        --real-dataset "argilla/10k_prompts_ranked_with_responses" \
        --generated-dataset "argilla/10k_prompts_SPIN_iter3_zephyr_top_generated" \
        --new-dataset "argilla/10k_prompts_SPIN_iter3_zephyr_top"
    ```

</details>


### Experiment *bottom* prompts

This contains the scripts to generate the same experiment from the *top* prompts, in this case selecting the *bottom* prompts (we have selected for those responses that have `num_response>1`, according to `avg_rating`, and selected the same amount of prompts that we had in the previous experiment, 1832).

The following are the steps to prepare the training data for SPIN, and the resulting datasets:

<details><summary> SPIN iter 0 </summary><hr>

- `generate_iter_spin.py`
    Script to generate the initial "generated" responses, from the SFT model that will then be fine-tuned 
    (**if the previous experiment was run, this dataset should be already generated**).

    Dataset: [argilla/10k_prompts_ranked_sft_zephyr](https://huggingface.co/datasets/argilla/10k_prompts_ranked_sft_zephyr)

    Run the following:

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
    Generates the dataset that will be directly ingested in the `SPINTrainer`.

    Dataset: [argilla/10k_prompts_bottom_SPIN_iter0](https://huggingface.co/datasets/argilla/10k_prompts_bottom_SPIN_iter0)

    Running the following python script: 

    ```console
    python prepare_for_training.py \
        --portion bottom \
        --target-dataset argilla/10k_prompts_SPIN_iter0_zephyr_bottom
    ```

</details>

<details><summary> SPIN iter 1 </summary><hr>

- `generate_iter_spin.py`

    Regenerates the "generated" responses from the model in the previous iteration:

    ```console
    python generate_iter_spin.py \
        --hf-apikey $HF_API_TOKEN \
        --source-dataset "argilla/10k_prompts_SPIN_iter0_zephyr_bottom" \
        --new-dataset "argilla/10k_prompts_SPIN_iter1_zephyr_bottom_generated" \
        --model-name "plaguss/zephyr-7b-spin-iter0-bottom-v0" \
        --batch-size 128 \
        --cuda-devices "0,1"
    ```

    Dataset: [argilla/10k_prompts_bottom_SPIN_iter1_generated](https://huggingface.co/datasets/argilla/10k_prompts_top_SPIN_iter1_generated)

- `transform_iter_generated.py`

    The script transforms the generated responses to the format expected by SPIN trainer:

    ```console
    python transform_iter_generated.py \
        --real-dataset "argilla/10k_prompts_ranked_with_responses" \
        --generated-dataset "argilla/10k_prompts_SPIN_iter1_zephyr_bottom_generated" \
        --new-dataset "argilla/10k_prompts_SPIN_iter1_zephyr_bottom"
    ```
</details>


<details><summary> SPIN iter 2 </summary><hr>

- `generate_iter_spin.py`

    Regenerates the "generated" responses from the model in the previous iteration:

    ```console
    python generate_iter_spin.py \
        --hf-apikey $HF_API_TOKEN \
        --source-dataset "argilla/10k_prompts_SPIN_iter1_zephyr_bottom" \
        --new-dataset "argilla/10k_prompts_SPIN_iter2_zephyr_bottom_generated" \
        --model-name "plaguss/zephyr-7b-spin-iter1-bottom-v0" \
        --batch-size 128 \
        --cuda-devices "0,1"
    ```

    Dataset: [argilla/10k_prompts_bottom_SPIN_iter1_generated](https://huggingface.co/datasets/argilla/10k_prompts_top_SPIN_iter1_generated)

- `transform_iter_generated.py`

    The script transforms the generated responses to the format expected by SPIN trainer:

    ```console
    python transform_iter_generated.py \
        --real-dataset "argilla/10k_prompts_ranked_with_responses" \
        --generated-dataset "argilla/10k_prompts_SPIN_iter2_zephyr_bottom_generated" \
        --new-dataset "argilla/10k_prompts_SPIN_iter2_zephyr_bottom"
    ```

</details>

<details><summary> SPIN iter 3 </summary><hr>

- `generate_iter_spin.py`

    Regenerates the "generated" responses from the model in the previous iteration:

    ```console
    python generate_iter_spin.py \
        --hf-apikey $HF_API_TOKEN \
        --source-dataset "argilla/10k_prompts_SPIN_iter2_zephyr_bottom" \
        --new-dataset "argilla/10k_prompts_SPIN_iter3_zephyr_bottom_generated" \
        --model-name "plaguss/zephyr-7b-spin-iter2-bottom-v0" \
        --batch-size 128 \
        --cuda-devices "0,1"
    ```

    Dataset: [argilla/10k_prompts_bottom_SPIN_iter1_generated](https://huggingface.co/datasets/argilla/10k_prompts_top_SPIN_iter1_generated)

- `transform_iter_generated.py`

    The script transforms the generated responses to the format expected by SPIN trainer:

    ```console
    python transform_iter_generated.py \
        --real-dataset "argilla/10k_prompts_ranked_with_responses" \
        --generated-dataset "argilla/10k_prompts_SPIN_iter3_zephyr_bottom_generated" \
        --new-dataset "argilla/10k_prompts_SPIN_iter3_zephyr_bottom"
    ```

</details>

### Experiment *random* prompts

This contains the scripts to generate the same experiment from the *top* prompts, in this case selecting a subset of *random* prompts (we have selected for those responses that have `num_response>1` 1832 records).

The following are the steps to prepare the training data for SPIN, and the resulting datasets:

<details><summary> SPIN iter 0 </summary><hr>

- `generate_iter_spin.py`
    Script to generate the initial "generated" responses, from the SFT model that will then be fine-tuned 
    (**if the previous experiment was run, this dataset should be already generated**).

    Dataset: [argilla/10k_prompts_ranked_sft_zephyr](https://huggingface.co/datasets/argilla/10k_prompts_ranked_sft_zephyr)

    Run the following:

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
    Generates the dataset that will be directly ingested in the `SPINTrainer`.

    Dataset: [argilla/10k_prompts_random_SPIN_iter0](https://huggingface.co/datasets/argilla/10k_prompts_random_SPIN_iter0)

    Running the following python script: 

    ```console
    python prepare_for_training.py \
        --portion random \
        --target-dataset argilla/10k_prompts_SPIN_iter0_zephyr_random
    ```

</details>

<details><summary> SPIN iter 1 </summary><hr>

- `generate_iter_spin.py`

    Regenerates the "generated" responses from the model in the previous iteration:

    ```console
    python generate_iter_spin.py \
        --hf-apikey $HF_API_TOKEN \
        --source-dataset "argilla/10k_prompts_SPIN_iter0_zephyr_random" \
        --new-dataset "argilla/10k_prompts_SPIN_iter1_zephyr_random_generated" \
        --model-name "plaguss/zephyr-7b-spin-iter0-random-v0" \
        --batch-size 128 \
        --cuda-devices "0,1"
    ```

    Dataset: [argilla/10k_prompts_bottom_SPIN_iter1_generated](https://huggingface.co/datasets/argilla/10k_prompts_top_SPIN_iter1_generated)

- `transform_iter_generated.py`

    The script transforms the generated responses to the format expected by SPIN trainer:

    ```console
    python transform_iter_generated.py \
        --real-dataset "argilla/10k_prompts_ranked_with_responses" \
        --generated-dataset "argilla/10k_prompts_SPIN_iter1_zephyr_random_generated" \
        --new-dataset "argilla/10k_prompts_SPIN_iter1_zephyr_random"
    ```
</details>

<details><summary> SPIN iter 2 </summary><hr>

- `generate_iter_spin.py`

    Regenerates the "generated" responses from the model in the previous iteration:

    ```console
    python generate_iter_spin.py \
        --hf-apikey $HF_API_TOKEN \
        --source-dataset "argilla/10k_prompts_SPIN_iter0_zephyr_random" \
        --new-dataset "argilla/10k_prompts_SPIN_iter2_zephyr_random_generated" \
        --model-name "plaguss/zephyr-7b-spin-iter2-random-v0" \
        --batch-size 128 \
        --cuda-devices "0,1"
    ```

    Dataset: [argilla/10k_prompts_bottom_SPIN_iter1_generated](https://huggingface.co/datasets/argilla/10k_prompts_top_SPIN_iter1_generated)

- `transform_iter_generated.py`

    The script transforms the generated responses to the format expected by SPIN trainer:

    ```console
    python transform_iter_generated.py \
        --real-dataset "argilla/10k_prompts_ranked_with_responses" \
        --generated-dataset "argilla/10k_prompts_SPIN_iter2_zephyr_random_generated" \
        --new-dataset "argilla/10k_prompts_SPIN_iter2_zephyr_random"
    ```
</details>

<details><summary> SPIN iter 3 </summary><hr>

- `generate_iter_spin.py`

    Regenerates the "generated" responses from the model in the previous iteration:

    ```console
    python generate_iter_spin.py \
        --hf-apikey $HF_API_TOKEN \
        --source-dataset "argilla/10k_prompts_SPIN_iter0_zephyr_random" \
        --new-dataset "argilla/10k_prompts_SPIN_iter3_zephyr_random_generated" \
        --model-name "plaguss/zephyr-7b-spin-iter3-random-v0" \
        --batch-size 128 \
        --cuda-devices "0,1"
    ```

    Dataset: [argilla/10k_prompts_bottom_SPIN_iter1_generated](https://huggingface.co/datasets/argilla/10k_prompts_top_SPIN_iter1_generated)

- `transform_iter_generated.py`

    The script transforms the generated responses to the format expected by SPIN trainer:

    ```console
    python transform_iter_generated.py \
        --real-dataset "argilla/10k_prompts_ranked_with_responses" \
        --generated-dataset "argilla/10k_prompts_SPIN_iter3_zephyr_random_generated" \
        --new-dataset "argilla/10k_prompts_SPIN_iter3_zephyr_random"
    ```
</details>

### Experiment *top* subset, Mistral-SFT-7B

<details><summary> SPIN iter 0 </summary><hr>

- `generate_iter_spin.py`
    Script to generate the initial "generated" responses, from the SFT model that will then be fine-tuned 

    Dataset: [argilla/10k_prompts_ranked_mistral_sft](https://huggingface.co/datasets/argilla/10k_prompts_ranked_mistral_sft)

    Run the following:

    ```console
    python generate_iter_spin.py \
        --hf-apikey $HF_API_TOKEN \
        --source-dataset "argilla/10k_prompts_SPIN_iter0_zephyr_top" \
        --new-dataset "argilla/10k_prompts_SPIN_iter0_mistral_sft_top_generated" \
        --model-name "HuggingFaceH4/mistral-7b-sft-beta" \
        --batch-size 128 \
        --cuda-devices "0,1"
    ```

- `prepare_for_training.py`
    Generates the dataset that will be directly ingested in the `SPINTrainer`.

    Dataset: [argilla/10k_prompts_random_SPIN_iter0](https://huggingface.co/datasets/argilla/10k_prompts_random_SPIN_iter0)

    Running the following python script: 

    ```console
    python transform_iter_generated.py \
        --real-dataset "argilla/10k_prompts_ranked_with_responses" \
        --generated-dataset "argilla/10k_prompts_SPIN_iter0_mistral_sft_top_generated" \
        --new-dataset "argilla/10k_prompts_SPIN_iter0_mistral_sft_top"
    ```

</details>

<details><summary> SPIN iter 1 </summary><hr>

- `generate_iter_spin.py`

    Regenerates the "generated" responses from the model in the previous iteration:

    ```console
    python generate_iter_spin.py \
        --hf-apikey $HF_API_TOKEN \
        --source-dataset "argilla/10k_prompts_SPIN_iter0_zephyr_random" \
        --new-dataset "argilla/10k_prompts_SPIN_iter1_zephyr_random_generated" \
        --model-name "plaguss/zephyr-7b-spin-iter0-random-v0" \
        --batch-size 128 \
        --cuda-devices "0,1"
    ```

    Dataset: [argilla/10k_prompts_bottom_SPIN_iter1_generated](https://huggingface.co/datasets/argilla/10k_prompts_top_SPIN_iter1_generated)

- `transform_iter_generated.py`

    The script transforms the generated responses to the format expected by SPIN trainer:

    ```console
    python transform_iter_generated.py \
        --real-dataset "argilla/10k_prompts_ranked_with_responses" \
        --generated-dataset "argilla/10k_prompts_SPIN_iter1_zephyr_random_generated" \
        --new-dataset "argilla/10k_prompts_SPIN_iter1_zephyr_random"
    ```
</details>

<details><summary> SPIN iter 2 </summary><hr>

- `generate_iter_spin.py`

    Regenerates the "generated" responses from the model in the previous iteration:

    ```console
    python generate_iter_spin.py \
        --hf-apikey $HF_API_TOKEN \
        --source-dataset "argilla/10k_prompts_SPIN_iter0_zephyr_random" \
        --new-dataset "argilla/10k_prompts_SPIN_iter2_zephyr_random_generated" \
        --model-name "plaguss/zephyr-7b-spin-iter2-random-v0" \
        --batch-size 128 \
        --cuda-devices "0,1"
    ```

    Dataset: [argilla/10k_prompts_bottom_SPIN_iter1_generated](https://huggingface.co/datasets/argilla/10k_prompts_top_SPIN_iter1_generated)

- `transform_iter_generated.py`

    The script transforms the generated responses to the format expected by SPIN trainer:

    ```console
    python transform_iter_generated.py \
        --real-dataset "argilla/10k_prompts_ranked_with_responses" \
        --generated-dataset "argilla/10k_prompts_SPIN_iter2_zephyr_random_generated" \
        --new-dataset "argilla/10k_prompts_SPIN_iter2_zephyr_random"
    ```
</details>

<details><summary> SPIN iter 3 </summary><hr>

- `generate_iter_spin.py`

    Regenerates the "generated" responses from the model in the previous iteration:

    ```console
    python generate_iter_spin.py \
        --hf-apikey $HF_API_TOKEN \
        --source-dataset "argilla/10k_prompts_SPIN_iter0_zephyr_random" \
        --new-dataset "argilla/10k_prompts_SPIN_iter3_zephyr_random_generated" \
        --model-name "plaguss/zephyr-7b-spin-iter3-random-v0" \
        --batch-size 128 \
        --cuda-devices "0,1"
    ```

    Dataset: [argilla/10k_prompts_bottom_SPIN_iter1_generated](https://huggingface.co/datasets/argilla/10k_prompts_top_SPIN_iter1_generated)

- `transform_iter_generated.py`

    The script transforms the generated responses to the format expected by SPIN trainer:

    ```console
    python transform_iter_generated.py \
        --real-dataset "argilla/10k_prompts_ranked_with_responses" \
        --generated-dataset "argilla/10k_prompts_SPIN_iter3_zephyr_random_generated" \
        --new-dataset "argilla/10k_prompts_SPIN_iter3_zephyr_random"
    ```
</details>


### Experiment *top* subset, Phi-2

From this point a single script is used instead. Run the `setup_full.sh` on a pod with 4 A100 80Gb GPUs, generate the different configuration files, and place the python script `generate_spin_dataset.py` under `SPIN/` folder once the repo has been downloaded, place `run_spin.sh` under `SPIN/scripts` folder, and run it:

```console
bash scripts/run_spin.sh
```

## Fine tune using SPIN

The following steps are almost a copy from the [SPIN](https://github.com/uclaml/SPIN) repository, take a look there for more information.

### Runpod

We used Runpod with the following setup:

- 4 A100 80Gb.
- 500Gb container and volume.
- Base image with CUDA 12.1.

### Once with the POD running

These are the steps outlined in the SPIN repo, you can run them by running the script in `scripts/setup.sh`:

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
```

And update the WANDB variables to keep track of the experiments:

```console
export WANDB_ENTITY="argilla-io"
export WANDB_PROJECT="dibt-spin-zephyr"
export WANDB_NAME="zephyr-7b-spin-iter1-v0"
```

After the previous step, replace the config file of the model to run, and the `finetune.sh` script, and start the training process:

```console
bash scripts/finetune.sh
```

### Weights and Biases runs

<details><summary> DIBT 10k *Top* subset </summary><hr>

- [argilla-io/dibt-top-spin-iter0-zephyr](https://wandb.ai/argilla-io/dibt-spin-zephyr/runs/439olh1m?nw=nwuserplagussargilla)

- [argilla-io/dibt-top-spin-iter1-zephyr](https://wandb.ai/argilla-io/dibt-spin-zephyr/runs/q938reyu?nw=nwuserplagussargilla)

- [argilla-io/dibt-top-spin-iter2-zephyr](https://wandb.ai/argilla-io/dibt-spin-zephyr/runs/q40amnp0?nw=nwuserplagussargilla)

- [argilla-io/dibt-top-spin-iter3-zephyr](https://wandb.ai/argilla-io/dibt-spin-zephyr/runs/u8znanpw?nw=nwuserplagussargilla)

</details>

<details><summary> DIBT 10k *Bottom* subset </summary><hr>

- [argilla-io/dibt-zephyr-7b-spin-iter0-bottom-v0](https://wandb.ai/argilla-io/dibt-spin-zephyr-bottom/runs/n9m0h7zq?nw=nwuserplagussargilla)

- [argilla-io/dibt-zephyr-7b-spin-iter1-bottom-v0](https://wandb.ai/argilla-io/dibt-spin-zephyr-bottom/runs/oj40r2bk?workspace=user-plaguss-argilla)

- [argilla-io/dibt-zephyr-7b-spin-iter2-bottom-v0](https://wandb.ai/argilla-io/dibt-spin-zephyr-bottom/runs/mjnmml88?nw=nwuserplagussargilla)

- [argilla-io/dibt-zephyr-7b-spin-iter3-bottom-v0](https://wandb.ai/argilla-io/dibt-spin-zephyr-bottom/runs/dgdm34mo?nw=nwuserplagussargilla)

</details>
