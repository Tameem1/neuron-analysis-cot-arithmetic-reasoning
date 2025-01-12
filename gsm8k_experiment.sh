#!/bin/bash -l
#SBATCH -J NEURON_ANALYSIS
#SBATCH -o run_out_3.txt
#SBATCH -p gpu-all
#SBATCH --gres=gpu:T4_16GB:4 
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 7:00:00 

# 1) Load modules
module load gcc11/11.3.0
module load cuda11.8/toolkit/11.8.0
module load cudnn8.9-cuda11.8/8.9.1.23

# 2) Activate conda environment
source ~/.bashrc
conda activate myexp

# 3) Move to the repository
cd ~/neuron-analysis-cot-arithmetic-reasoning

# 4) (Optional) If you need data in data/test.jsonl or gsm8k data, ensure it's in place

# 5) Run the GSM8K Inference
torchrun --nproc_per_node 4 gsm8k_inference.py --ckpt_dir ~/models/qwen2.5-math-1.5B-instruct --tokenizer_path ~/models/qwen2.5-math-1.5B-instruct/tokenizer.json --prompt data/prompts/equation_only.txt --few_shot False --results_dir results

# 6) Run Algorithm One
# torchrun --nproc_per_node=1 main.py \
#   --ckpt_dir /home/username/path_to_model \
#   --tokenizer_path /home/username/path_to_model \
#   --experiment algorithm_one \
#   --prompt data/prompts/equation_only.txt \
#   --data_dir results/gsm8k_inference/equation_only/final.json \
#   --results_dir results/algorithm_one/

# 7) Run Algorithm Two
# torchrun --nproc_per_node=1 main.py \
#   --ckpt_dir /home/username/path_to_model \
#   --tokenizer_path /home/username/path_to_model \
#   --experiment algorithm_two \
#   --prompt data/prompts/equation_only.txt \
#   --data_dir results/gsm8k_inference/equation_only/final.json \
#   --results_dir results/algorithm_two/

echo "GSM8K inference completed"