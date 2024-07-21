#!/bin/bash

#echo $(pwd)
unset PYTHONPATH
export PYTHONPATH=$(pwd):$(pwd)/src:$(pwd)/utils:$PYTHONPATH
