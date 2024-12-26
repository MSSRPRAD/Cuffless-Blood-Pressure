#!/bin/bash
#SBATCH -p gpu
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_a100_8
#SBATCH --mem 200G
#SBATCH -t 2-00:00
#SBATCH -o test.out
#SBATCH -e test.err
#SBATCH --mail-user=f20210367@hyderabad.bits-pilani.ac.in
#SBATCH --mail-type=ALL
# Load modules

#source virtual/bin/activate
#pip freeze
#conda activate octave
#octave
source .venv/bin/activate
which python3
srun python3 foo.py

# Activate Environment
#source
#echo "setting up conda....."
#conda init bash
#echo "Activating Environment....."
# conda env list
# conda activate icl
# conda info
#source venv/bin/activate
#pip freeze
#which python
#which python3
#conda install -n base conda-forge::mamba
#mamba install -y transformers
#mamba install -y pytorch torchvision -c pytorch
#mamba install wandb
# Run
# cd
#python data/circuits/prepare.py
#echo "contents of directory......"
#ls
#nvidia-smi
#echo "started training!"
#python3 experiments.py
