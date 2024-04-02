accelerate launch \
    --config_file configs/deepspeed-zero3.yaml \
    --num_processes 4 \
    --no_python train sft --config-path config_full.yaml