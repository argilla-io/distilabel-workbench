accelerate launch \
    --config_file configs/deepspeed-zero3.yaml \
    --num_processes 4 \
    --no_python train sft --config-path config_full.yaml

# Automatically remove the pod after finishing
runpodctl remove pod $RUNPOD_POD_ID