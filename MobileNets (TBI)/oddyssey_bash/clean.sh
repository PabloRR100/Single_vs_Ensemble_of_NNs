#!/bin/bash

# Remove previous errors
find . -type f -name myerrors\* -exec rm {} \;
find . -type f -name myoutput\* -exec rm {} \;

find . -type f -name slurm\* -exec rm {} \;

find . -type f -name test\* -exec rm {} \;

exit
