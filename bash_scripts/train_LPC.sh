#!/usr/bin/env bash

# Run from project's root

# Set PYTHONPATH
parentdir="$(dirname "$(pwd)")"
export PYTHONPATH=$PYTHONPATH:$parentdir

# Compute features
echo 'Start training'

declare -a algs=("z_score" "minmax_scaler" "l2_norm" "l1_norm" "max_norm" "no_norm")
declare -a cnns=("tutorial_cnn")

for a in "${algs[@]}"
do

  for c in  "${cnns[@]}"
  do
    python3 /nas/home/cborrelli/bot_speech/python_scripts/LPC_train.py --preprocessing "$a" --preprocessing_axis 0 --cnn "$c"
    python3 /nas/home/cborrelli/bot_speech/python_scripts/LPC_train.py --preprocessing "$a" --preprocessing_axis 2 --cnn "$c"
    python3 /nas/home/cborrelli/bot_speech/python_scripts/LPC_train.py --preprocessing "$a" --preprocessing_axis 0 2 --cnn "$c"

  done


done