#!/bin/bash

# Confuguration for multiple GPU on a single Machine

#SBATCH -n 1                # Number of cores
#SBATCH -N 1                # Ensure that all cores are on one machine
#SBATCH -t 0-00:30          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p gpu_requeue      # Partition to submit to

# Testing mode - quickly allocation of resources
#SBATCH --mem=30000          # Memory pool for all cores (see also --mem-per-cpu)

# Uncomment to full power
#SBATCH --gres=gpu:2        # Activate n GPU (let's say 8)

#SBATCH -o myoutput_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e myerrors_%j.err  # File to which STDERR will be written, %j inserts jobid

module load git/2.17.0-fasrc01
module load Anaconda3/5.0.1-fasrc02

# Check arguments have been passed
if [ $# -lt 2 ]
then

  check=0
  while [ $check -lt 1 ]
  do

    echo
    echo Please, insert cuda version. Insert [1 / 2]
    echo Options: [1] cuda80, [2] cuda90
    read cuda

    if [ $cuda -gt 0 ] && [ $cuda -lt 3 ]
    then
      let check=check+1
    else
      echo
      echo Insert a valid option!
      echo
    fi

  done

  if [ $cuda == 1 -o $cuda == 'cuda80' ]
  then
    cuda="cuda80"
  else
    cuda="cuda90"
  fi

  check=0
  while [ $check -lt 1 ]
  do

    echo
    echo Run testing mode? Insert [1 / 2]
    echo Options: [1] True, [2] False
    read testing

    if [ $testing -gt 0 ] && [ $testing -lt 3 ]
    then
      let check=check+1
    else
      echo
      echo Insert a valid option!
      echo
    fi

  done

  if [ $testing == 1 -o $testing == 'True' ]
  then
    testing=True
  else
    testing=False
  fi

fi

set $cuda $testing

echo
echo
echo Settings:
echo cuda: $1
echo testing: $2


if [ $1 == 'cuda80' ]
then
  echo Activating environment cuda8...
  module load cuda/8.0.61-fasrc01 cudnn/6.0_cuda8.0-fasrc01
  CONDA="cuda8"
elif [ $1 == 'cuda90' ]
then
  echo Activating environment torch37...
  module load cuda/9.0-fasrc02 cudnn/7.0_cuda9.0-fasrc01
  CONDA="torch37"
else
  echo Environment not found
  echo Please, rememeber to insert env as parameter
  echo Aborting...
  exit 1
fi

# Output information
manage_env () {
  source activate $CONDA
  which python
  python -V
}

# Launch notebook
manage_env
echo Loading Script...

python demo_CIFAR10.py --name ResNet --save False --testing $2 --comments True --draws False --ensembleSize Big --dataset fruits-360-small
# python demo_CIFAR10.py --name  --save True --﻿testing False --comments True --draws False --ensembleSize Big --dataset CIFAR10

exit
