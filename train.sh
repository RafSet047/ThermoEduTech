#!/bin/bash

configs_dirpath=$1

for config in $configs_dirpath/* ; do
    python src/train.py -c $config
done