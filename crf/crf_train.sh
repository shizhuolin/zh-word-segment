#!/bin/sh
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --nodelist=linux9
srun ./pku-test-crf.sh
