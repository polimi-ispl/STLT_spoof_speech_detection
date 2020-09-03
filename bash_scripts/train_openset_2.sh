#!/usr/bin/env bash

# Run from project's root

# Set PYTHONPATH
parentdir="$(dirname "$(pwd)")"
export PYTHONPATH=$PYTHONPATH:$parentdir

# Compute features
echo 'Run script 2'
python3 /nas/home/cborrelli/bot_speech/python_scripts/train_openset.py --nfft 128 --hop_size 64 --classifiers rf svm --normalizers l2
python3 /nas/home/cborrelli/bot_speech/python_scripts/train_openset.py --nfft 256 --hop_size 128 --classifiers rf svm --normalizers l2
python3 /nas/home/cborrelli/bot_speech/python_scripts/train_openset.py --nfft 512 --hop_size 256 --classifiers rf svm --normalizers l2
