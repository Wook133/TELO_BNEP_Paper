#!/bin/bash
#PBS -l select=1:ncpus=24:mpiprocs=24:nodetype=haswell_reg
#PBS -P CSCI1305
#PBS -q serial
#PBS -l walltime=36:00:00
#PBS -o /mnt/lustre/users/jdevilliers1/november2020/ppo_output/test_3_dec/ppo_pyant_test_000.out
#PBS -e /mnt/lustre/users/jdevilliers1/november2020/ppo_output/test_3_dec/ppo_pyant_test_000.err
#PBS -m abe
#PBS -M -redacted-
#PBS -S /bin/bash
module purge
module add gcc/6.1.0
module add chpc/python/anaconda/3
module add chpc/python/anaconda/3-2019.10
module add chpc/python/anaconda/3-2020.02
module load gcc/6.1.0 chpc/python/anaconda/3 chpc/python/anaconda/3-2020.02 chpc/python/anaconda/3-2019.10
conda info --envs
conda activate /home/jdevilliers1/earl
conda list
cd /home/jdevilliers1/lustre/november2020
python bohb_ppo_pyant_tuned.py --shared_directory /mnt/lustre/users/jdevilliers1/november2020/ppo_output/test_3_dec/ --num_processors 24
conda deactivate
