#!/bin/bash -l
#SBATCH -J gsm8k_inference_array                # Job name
#SBATCH -p gpu-short                            # Partition
#SBATCH --gres=gpu:T4_16GB                      # 1 GPU per sub-job
#SBATCH -c 4                                    # CPU cores per sub-job
#SBATCH --mem=16G                               # Memory per sub-job
#SBATCH --array=5,6,7,8,9                         # Remaining chunks
#SBATCH -o logs/%A_%a.out                       # Log file (%A = array job ID, %a = sub-job index)

# (Optional) set a max run time if needed:
## #SBATCH --time=01:00:00

module load cuda11.8/toolkit                    # or whichever CUDA module is right for your code
module load cudnn8.9-cuda11.8/8.9.1.23          # If you need a particular cuDNN
source ~/.bashrc
conda activate myexp                            # or your environment name

# We'll use $SLURM_ARRAY_TASK_ID to pick which chunk to run on.
CHUNK_ID=${SLURM_ARRAY_TASK_ID}

# Point to the chunk you want:
DATA_PATH="data/data_chunk_${CHUNK_ID}.jsonl"

# Decide how you want to name the output directory
RESULTS_PATH="results/chunk_${CHUNK_ID}"

echo "Processing data chunk: $DATA_PATH"
echo "Saving results to:     $RESULTS_PATH"

# Now run your gsm8k_inference code
torchrun --nproc_per_node=1 gsm8k_inference.py \
  --ckpt_dir ~/models/qwen2.5-math-1.5B-instruct \
  --tokenizer_path ~/models/qwen2.5-math-1.5B-instruct/tokenizer.json \
  --prompt data/prompts/cot_prompt_1shot.txt \
  --few_shot True \
  --results_dir "$RESULTS_PATH" \
  --data_path "$DATA_PATH"