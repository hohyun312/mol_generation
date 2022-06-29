#!/bin/bash

#SBATCH --job-name=DND           # Submit a job named "example"
#SBATCH --nodes=1                # Using 1 node
#SBATCH --gres=gpu:0             # Using 1 GPU
#SBATCH --time=0-12:00:00        # 12 hours timelimit
#SBATCH --mem=16000MB            # Using 16GB memory
#SBATCH --cpus-per-task=2        # Using 8 cpus per task (srun) 
#SBATCH --output=slurm.log       # Creating log file

source /home/hohyunkim/.bashrc
source /home/hohyunkim/anaconda/etc/profile.d/conda.sh
conda activate rdkit06

echo "BASH START"
srun python ppo_train.py
echo "BASH END"