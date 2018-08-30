#!/bin/bash

module load git/2.17.0-fasrc01
module load Anaconda3/5.0.1-fasrc02

if [ $1 == 'cuda80' ]
then
  module load cuda/8.0.61-fasrc01 cudnn/6.0_cuda8.0-fasrc0
  source activate cuda80

elif [ $1 == 'cuda90' ]
then
  module load cuda/9.0-fasrc02 cudnn/7.0_cuda9.0-fasrc01
  source activate torch37
fi

# Output information
which python
python -V

# Launch notebook
jupyter notebook --port 8899
