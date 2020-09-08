#!/usr/bin/env bash

# Run from project's root

# Set PYTHONPATH
parentdir="$(dirname "$(pwd)")"
export PYTHONPATH=$PYTHONPATH:$parentdir

# Compute features
echo 'Run script 4'

echo "Bicoh"
echo "NFFT 128"
python3 /nas/home/cborrelli/bot_speech/python_scripts/train_openset.py --nfft 128 --hop_size 64 --classifiers svm --normalizers minmax zscore l2 --selected_features bicoh
echo "NFFT 256"
python3 /nas/home/cborrelli/bot_speech/python_scripts/train_openset.py --nfft 256 --hop_size 128 --classifiers svm --normalizers minmax zscore l2 --selected_features bicoh
echo "NFFT 512"
python3 /nas/home/cborrelli/bot_speech/python_scripts/train_openset.py --nfft 512 --hop_size 256 --classifiers svm --normalizers minmax zscore l2 --selected_features bicoh
