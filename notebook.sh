#!/bin/bash

module load git/2.17.0-fasrc01
module load Anaconda3/5.0.1-fasrc02
module load cuda/9.0-fasrc02 cudnn/7.0_cuda9.0-fasrc01

# Activate Environment
source activate torch37

which python
python -V

# Launch notebook
jupyter notebook --port 8899
 