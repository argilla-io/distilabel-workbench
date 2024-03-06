accelerate launch \
    --config_file configs/multi-gpu.yaml \
    --num_processes 1 \
    --no_python train dpo \
    --config-path config-lora.yaml