#!/usr/bin/env bash

# Run from project's root

# Set PYTHONPATH
parentdir="$(dirname "$(pwd)")"
export PYTHONPATH=$PYTHONPATH:$parentdir

# Compute features
echo 'Run closed set binary'
python3 /nas/home/cborrelli/bot_speech/python_scripts/train_closedset_binary.py --nfft 128 --hop_size 64 --classifiers svm --normalizers zscore minmax l2 --selected_features lpc
#python3 /nas/home/cborrelli/bot_speech/python_scripts/train_closedset_binary.py --nfft 256 --hop_size 128 --classifiers svm --normalizers zscore  minmax l2 --selected_features lpc
#python3 /nas/home/cborrelli/bot_speech/python_scripts/train_closedset_binary.py --nfft 512 --hop_size 256 --classifiers svm --normalizers zscore minmax l2 --selected_features lpc

#python3 /nas/home/cborrelli/bot_speech/python_scripts/train_closedset_binary.py --nfft 128 --hop_size 64 --classifiers svm --normalizers zscore minmax l2 --selected_features unet
#python3 /nas/home/cborrelli/bot_speech/python_scripts/train_closedset_binary.py --nfft 256 --hop_size 128 --classifiers svm --normalizers zscore zscore minmax l2 --selected_features unet
#python3 /nas/home/cborrelli/bot_speech/python_scripts/train_closedset_binary.py --nfft 512 --hop_size 256 --classifiers svm --normalizers zscore zscore minmax l2 --selected_features unet

#python3 /nas/home/cborrelli/bot_speech/python_scripts/train_closedset_binary.py --nfft 128 --hop_size 64 --classifiers svm --normalizers zscore minmax l2 --selected_features lpc unet
#python3 /nas/home/cborrelli/bot_speech/python_scripts/train_closedset_binary.py --nfft 256 --hop_size 128 --classifiers svm --normalizers zscore  minmax l2 --selected_features lpc unet
#python3 /nas/home/cborrelli/bot_speech/python_scripts/train_closedset_binary.py --nfft 512 --hop_size 256 --classifiers svm --normalizers zscore minmax l2 --selected_features lpc unet