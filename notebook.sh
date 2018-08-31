#!/bin/bash

module load git/2.17.0-fasrc01
module load Anaconda3/5.0.1-fasrc02

echo type: $1
echo cuda: $2

if [ $2 == 'cuda80' ]
then
  echo Activating environment cuda8...
  module load cuda/8.0.61-fasrc01 cudnn/6.0_cuda8.0-fasrc01
  CONDA="cuda8"

elif [ $2 == 'cuda90' ]
then
  echo Activating environment torch37...
  module load cuda/9.0-fasrc02 cudnn/7.0_cuda9.0-fasrc01
  ENV="torch37"

else
  echo Environment not found
  echo Please, rememeber to insert env as parameter
  echo Aborting...
  exit 1
fi

# Output information
source activate $CONDA
which python
python -V

# Launch notebook
if [ $1 == 'notebook' ]
then
  echo Loading Jupyer Notebook ...
  jupyter notebook --port 8899
else
  echo Loading Jupyter Lab ...
  jupyter lab --port 8899
fi
