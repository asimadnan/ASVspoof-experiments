#!/bin/bash
#PBS -P fk99
#PBS -q gpuvolta
#PBS -l ngpus=4
#PBS -l ncpus=48
#PBS -l mem=382G
#PBS -l walltime=1:30:00
#PBS -l wd
#PBS -l jobfs=20GB
#PBS -l storage=scratch/fk99

# Load modules
module purge
#module load python3/3.8.5
module load python3/3.9.2
module load tensorflow/2.4.1
module load ffmpeg/4.3.1
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:~/libsndfile/build


#cqcc_300_spoof
python3 alexnet2.py
python3 alexnet-eval.py