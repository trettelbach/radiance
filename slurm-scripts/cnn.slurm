#! /bin/bash -l

#SBATCH --ntasks=1
#SBATCH --time=23:59:00
#SBATCH --account=<ACCOUNT>
#SBATCH --export=NONE
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:4
#SBATCH --constraint=p100
#SBATCH --job-name=cnn

module load shifter

shifter run trettelbach/pytorch_astro:latest python <PATH/TO/PYTHONFILE.py> <'PATH/TO/DIRECTORY/OF/CSV-FILE/'> <STATS>
