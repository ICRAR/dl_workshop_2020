#!/bin/bash

#SBATCH --job-name="GZHv1PSP4"

## debugging
###########SBATCH --nodes=1
###########SBATCH -t 00:20:00
###########SBATCH --cpus-per-task=5
###########SBATCH --gres=gpu:2
###########SBATCH --ntasks-per-node=2 ##### This should be EQUAL to the number of GPUs for the MPI, specifiying the gres=gpu:4 only doesn't work
###########SBATCH --mem=24gb ### It requires much more memory due to the more layers in the list (I think!) 


#SBATCH -x b031

### True run
#SBATCH --nodes=1
#SBATCH -t 23:59:00
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4
######## #SBATCH --ntasks-per-node=4 ##### This should be EQUAL to the number of GPUs for the MPI, specifiying the gres=gpu:4 only doesn't work
#SBATCH --mem=16gb ### It requires much more memory due to the more layers in the list (I think!) 



##### Number of total processes 
echo " "
echo " Nodelist:= " $SLURM_JOB_NODELIST
echo " Number of nodes:= " $SLURM_JOB_NUM_NODES
echo " NGPUs per node:= " $SLURM_GPUS_PER_NODE 
echo " "

source ~/.bashrc 

module list


echo "Run started at:- "
date

python train_x_parallel.py


echo "Run completed at:- "
date
