#!/bin/bash			 
# COMMENT:			 
#SBATCH --job-name=segment_train
#SBATCH --mail-user=mawais@umich.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --cpus-per-task=24
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=1000m
#SBATCH --time=24:00:00
#SBATCH --account=drmalik0
#SBATCH --partition=gpu		  # standard, largemem, spgpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-gpu=1
#SBATCH --output=/home/mawais/Specnet/%x-%j.log
# COMMENT:The application(s) to execute along with its input arguments and options:


cd /home/mawais/notebooks

source /sw/pkgs/arc/python3.9-anaconda/2021.11/etc/profile.d/conda.sh
conda activate pytorch_env

python3 Segment_train_whc.py