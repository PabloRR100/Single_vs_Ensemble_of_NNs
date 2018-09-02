#!/bin/bash

# Confuguration for multiple GPU on a single Machine

#SBATCH -n 1                # Number of cores
#SBATCH -N 1                # Ensure that all cores are on one machine
#SBATCH -t 0-00:10          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p gpu_requeue      # Partition to submit to

# Testing mode - quickly allocation of resources
#SBATCH --mem=30000          # Memory pool for all cores (see also --mem-per-cpu)

# Uncomment to full power
#SBATCH --gres=gpu:2        # Activate n GPU (let's say 8)

#SBATCH -o myoutput_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e myerrors_%j.err  # File to which STDERR will be written, %j inserts jobid

module load git/2.17.0-fasrc01          # Load Git
module load Anaconda3/5.0.1-fasrc02     # Load Anaconda

# Pass the CUDA version as argument
if [ $1 == 'cuda80' ]
then
  echo Activating environment cuda8
  module load cuda/8.0.61-fasrc01 cudnn/6.0_cuda8.0-fasrc01
  source activate cuda8

elif [ $1 == 'cuda90' ]
then
  echo Activating environment torch37
  module load cuda/9.0-fasrc02 cudnn/7.0_cuda9.0-fasrc01
  source activate torch37

else
  echo ENVIRONMENT NOT LOADED
  echo Exiting...
  exit 1
fi


# ResNet CIFAR10
# --------------

# Big Ensemble
# if [$2 == 'big']


python demo_CIFAR10.py --name ResNet --save True --testing False --comments True --draws False --ensembleSize Big --dataset CIFAR10 
 