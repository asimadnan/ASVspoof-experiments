#!/bin/bash
#PBS -P fk99
#PBS -q hugemem
#PBS -l ncpus=96
#PBS -l mem=256G
#PBS -l walltime=00:40:00
#PBS -l wd
#PBS -l jobfs=100GB
#PBS -l storage=scratch/fk99

# Load modules
module purge
#module load python3/3.8.5
module load python3/3.9.2
module load tensorflow/2.4.1
module load ffmpeg/4.3.1
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:~/libsndfile/build


#cqcc_300_spoof
python3 alexnet-multip.py train amplitutde
python3 alexnet-multip.py dev amplitutde