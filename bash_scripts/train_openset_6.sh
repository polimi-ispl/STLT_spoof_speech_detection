#!/usr/bin/env bash

# Run from project's root

# Set PYTHONPATH
parentdir="$(dirname "$(pwd)")"
export PYTHONPATH=$PYTHONPATH:$parentdir

# Compute features
echo 'Run script 6'
echo "LPC"
python3 /nas/home/cborrelli/bot_speech/python_scripts/train_openset.py --nfft 128 --hop_size 64 --classifiers svm --normalizers minmax zscore l2 --selected_features lpc

