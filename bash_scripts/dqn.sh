#!/bin/bash

#PBS -q dssc_gpu
#PBS -l nodes=1:ppn=1
#PBS -l walltime=120:00:00

cd $PBS_O_WORKDIR

~/.conda/envs/atari/bin/python -m atari_py.import_roms /u/dssc/s271711/Atari/Utils

~/.conda/envs/atari/bin/python /u/dssc/s271711/Atari/Code/DQN.py
