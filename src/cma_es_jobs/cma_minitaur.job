#!/bin/bash
#PBS -l select=1:ncpus=23:mpiprocs=23:nodetype=haswell_reg
#PBS -P CSCI1305
#PBS -q serial
#PBS -l walltime=09:30:00
#PBS -o /mnt/lustre/users/jdevilliers1/november2020/es_output/cma_minitaur_14.out
#PBS -e /mnt/lustre/users/jdevilliers1/november2020/es_output/cma_minitaur_14.err
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
python cmaes_rl.py --env_name MinitaurBulletEnv-v0 --max_num_episodes 38400 --population_size 256 --rank_fitness 0 --num_processors 23 --num_evaluations 1 --sigma_init 0.0617566507403939 --mu_init -0.00869837283010777 --weight_decay 0.238433123453988 --use_default_layers 0 --use_max_action 0 --shared_directory /mnt/lustre/users/jdevilliers1/november2020/es_output/redo/
conda deactivate

