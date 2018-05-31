#!/bin/sh
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --nodelist=linux[7-9]
srun ./pku-test-crf.sh 10 1
