#!/bin/bash

# Confuguration for multiple GPU on a single Machine

#SBATCH -p gpu_requeue      # Partition to submit to
#SBATCH -t 0-00:30          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH --mem=30000          # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --gres=gpu:2        # Activate n GPU (let's say 8)

#SBATCH --tunnel 8899:8899  # Open tunnel Compute_Host - Login_Host

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
    echo Please, insert mode. Insert [1 / 2]
    echo Options: [1] Notebook, [2] Lab
    read mode

    if [ $mode -gt 0 ] && [ $mode -lt 3 ]
    then
      let check=check+1
    else
      echo
      echo Insert a valid option!
      echo
    fi

  done

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

fi

echo mode: $mode
echo cuda: $cuda

if [ $cuda == 'cuda80' ]
then
  echo Activating environment cuda8...
  module load cuda/8.0.61-fasrc01 cudnn/6.0_cuda8.0-fasrc01
  CONDA="cuda8"

elif [ $cuda == 'cuda90' ]
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
if [ $mode == 'notebook' ]
then
  manage_env
  echo Loading Jupyter Notebook ...
  jupyter notebook --port 8899
else
  manage_env
  echo Loading Jupyter Lab ...
  jupyter lab --port 8899
fi
