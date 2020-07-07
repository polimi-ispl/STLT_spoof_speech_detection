#!/usr/bin/env bash

# Run from project's root

# Set PYTHONPATH
parentdir="$(dirname "$(pwd)")"
export PYTHONPATH=$PYTHONPATH:$parentdir

# Compute features
echo 'Compute Features'

for i in {1..50}
do

    python3 /nas/home/cborrelli/bot_speech/python_scripts/compute_LPC_features.py --audio_folder /nas/home/cborrelli/bot_speech/dataset/LA/ASVspoof2019_LA_train/flac --txt_path /nas/home/cborrelli/bot_speech/dataset/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt --dest_folder /nas/home/cborrelli/bot_speech/features/lpc --lpc_length $i --data_subset train

    python3 /nas/home/cborrelli/bot_speech/python_scripts/compute_LPC_features.py --audio_folder /nas/home/cborrelli/bot_speech/dataset/LA/ASVspoof2019_LA_dev/flac --txt_path /nas/home/cborrelli/bot_speech/dataset/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt --dest_folder /nas/home/cborrelli/bot_speech/features/lpc --lpc_length $i --data_subset dev

    python3 /nas/home/cborrelli/bot_speech/python_scripts/compute_LPC_features.py --audio_folder /nas/home/cborrelli/bot_speech/dataset/LA/ASVspoof2019_LA_eval/flac --txt_path /nas/home/cborrelli/bot_speech/dataset/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt --dest_folder /nas/home/cborrelli/bot_speech/features/lpc --lpc_length $i --data_subset eval

done