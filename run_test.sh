#!/bin/bash
#SBATCH --job-name=lm_test        # Job name
#SBATCH --output=lm_test.out      # Standard output and error log
#SBATCH --ntasks=1                # Number of tasks (processes)
#SBATCH --cpus-per-task=4         # Number of CPU cores per task
#SBATCH --mem=16G                 # Total memory per node
#SBATCH --time=01:00:00           # Time limit hrs:min:sec
#SBATCH -p gpu-short

# 1) Load modules
module load gcc11/11.3.0
module load cuda11.8/toolkit/11.8.0
module load cudnn8.9-cuda11.8/8.9.1.23

# 2) Activate conda environment
source ~/.bashrc
conda activate myexp

# 3) Move to the repository
cd ~/neuron-analysis-cot-arithmetic-reasoning

# Ensure all dependencies are installed
# Optionally, uncomment the following lines to install dependencies
# pip install --upgrade pip
# pip install transformers torch sentencepiece

# Run the test script
python test_model.py

# Deactivate the virtual environment (optional)
deactivate