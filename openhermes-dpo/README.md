## OpenHermes-dpo

This folder contains the script used to generate a part of the dataset from `teknium/OpenHermes-2.5` using `NousResearch/Nous-Hermes-2-Yi-34B`.

### Notes:

- It was run in runpod using 4xA100 GPUs of 80Gb (Using CUDA 12).

- For a batch size of 512, it took close to 3 minutes per batch to complete (this parameter still needs more investigation within `vllm`, could have taken a bigger size? how to estimate the optimum value with this framework?).

- The argument `tensor_parallel_size` from `vllm` must be divisible by the number of *attention heads* of the model. `NousResearch/Nous-Hermes-2-Yi-34B` has 56, so 2, 4 or 8 GPUs would work, updating the parameter accordingly.

*Due to time constraints, the dataset was generated in different parts, and the notebook contains the extra work to merge the different datasets and keep the index related to the original dataset*.