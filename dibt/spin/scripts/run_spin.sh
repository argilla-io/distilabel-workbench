# Set which GPU devices to be visible to the process, --num_processes should be adjusted accordingly
export CUDA_VISIBLE_DEVICES="0,1,2,3"

# Set the logging level for the `accelerate` library to output informational messages.
ACCELERATE_LOG_LEVEL=info
export WANDB_ENTITY="argilla-io"
# Update the project name to the one you will use
export WANDB_PROJECT="dibt-spin-mistral-sft"

# The following are the config files for the different models to train,
# these must be already defined and in the configs/ folder from the SPIN repo
config_files=(
    "configs/config_iter0_small.yaml"
    "configs/config_iter1_small.yaml"
    "configs/config_iter2_small.yaml"
    "configs/config_iter3_small.yaml"
)
# The model from which you want to start the SPIN iterations,
# will be used for the first dataset generation and fine tune.
initial_model="microsoft/phi-2"
# Base model name for the different SPIN iterations, the models will be named as:
# - base_model_name{_iter{i}}
base_model_name="plaguss/phi-2-spin-top-v0"
# The dataset that will be used in the first iteration to generate the responses.
# It must contain a column with the name "prompt" from which the responses
# will be generated
initial_dataset="argilla/10k_prompts_SPIN_iter0_zephyr_top"
# Base model name for the different dataset SPIN iterations, the datasets will be named as:
# - base_dataset_name{_iter{i}}
base_dataset_name="argilla/10k_prompts_SPIN_phi2_top"
# The dataset from which the subsequent generations will be obtained, will
# be a copy of base_dataset_name, and will update iteratively
source_dataset_base=base_dataset_name
# The dataset containing the reference responses
real_dataset="argilla/10k_prompts_ranked_with_responses"
# The base name for the WANDB experiments, under WANDB_PROJECT
wandb_name_base="spin-phi2"

for ((i = 0; i < ${#config_files[@]}; i++)); do
    if [ "$i" = "0" ]; then
        # Model for the first iteration
        model_name_for_generation="${initial_model}"
        source_dataset="${initial_dataset}"
    else
        # Model updated on each iteration
        model_name_for_generation="${base_model_name}_iter${i-1}"
        source_dataset="${source_dataset_base}_iter${i-1}"
    fi

    new_dataset="${base_dataset_name}_iter$i"
    command_dataset_generation="python generate_spin_dataset.py \
        --hf-apikey $HF_API_TOKEN \
        --source-dataset ${source_dataset} \
        --new-dataset ${new_dataset} \
        --real-dataset ${real_dataset} \
        --model-name $model_name_for_generation{} \
        --batch-size 512 \
        --cuda-devices '0,1,2,3'"
    
    eval $command_dataset_generation
    
    export WANDB_NAME="${wandb_name_base}_iter$i"
    command_train="accelerate launch --config_file configs/deepspeed_zero3.yaml --num_processes=4 spin/run_spin.py configs/${config_files[i]}.yaml"
    
    eval $command_train

    echo "----- Iter$i finished -----"

done

echo "---- Process finished ----"