#!/usr/bin/env bash

# Run from project's root

# Set PYTHONPATH
parentdir="$(dirname "$(pwd)")"
export PYTHONPATH=$PYTHONPATH:$parentdir

# Compute features
echo 'Compute Features for train set'

python3 python_scripts/compute_spectrogram_features.py  \
--audio_folder /nas/home/cborrelli/bot_speech/dataset/LA/ASVspoof2019_LA_train/flac \
--txt_path /nas/home/cborrelli/bot_speech/dataset/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt \
--dest_folder features/spectrogram \
--nfft 512 --hop_size 256 --data_subset train

python3 python_scripts/compute_spectrogram_features.py  \
--audio_folder /nas/home/cborrelli/bot_speech/dataset/LA/ASVspoof2019_LA_train/flac \
--txt_path /nas/home/cborrelli/bot_speech/dataset/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt \
--dest_folder features/spectrogram \
--nfft 256 --hop_size 128 --data_subset train

python3 python_scripts/compute_spectrogram_features.py  \
--audio_folder /nas/home/cborrelli/bot_speech/dataset/LA/ASVspoof2019_LA_train/flac \
--txt_path /nas/home/cborrelli/bot_speech/dataset/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt \
--dest_folder features/spectrogram \
--nfft 128 --hop_size 64 --data_subset train

python3 python_scripts/compute_spectrogram_features.py  \
--audio_folder /nas/home/cborrelli/bot_speech/dataset/LA/ASVspoof2019_LA_train/flac \
--txt_path /nas/home/cborrelli/bot_speech/dataset/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt \
--dest_folder features/spectrogram \
--nfft 64 --hop_size 32 --data_subset train

echo 'Compute Features for dev set'

python3 python_scripts/compute_spectrogram_features.py  \
--audio_folder /nas/home/cborrelli/bot_speech/dataset/LA/ASVspoof2019_LA_dev/flac \
--txt_path /nas/home/cborrelli/bot_speech/dataset/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt \
--dest_folder features/spectrogram \
--nfft 512 --hop_size 256 --data_subset dev

python3 python_scripts/compute_spectrogram_features.py  \
--audio_folder /nas/home/cborrelli/bot_speech/dataset/LA/ASVspoof2019_LA_dev/flac \
--txt_path /nas/home/cborrelli/bot_speech/dataset/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt \
--dest_folder features/spectrogram \
--nfft 256 --hop_size 128 --data_subset dev

python3 python_scripts/compute_spectrogram_features.py  \
--audio_folder /nas/home/cborrelli/bot_speech/dataset/LA/ASVspoof2019_LA_dev/flac \
--txt_path /nas/home/cborrelli/bot_speech/dataset/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt \
--dest_folder features/spectrogram \
--nfft 128 --hop_size 64 --data_subset dev

python3 python_scripts/compute_spectrogram_features.py  \
--audio_folder /nas/home/cborrelli/bot_speech/dataset/LA/ASVspoof2019_LA_dev/flac \
--txt_path /nas/home/cborrelli/bot_speech/dataset/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt \
--dest_folder features/spectrogram \
--nfft 64 --hop_size 32 --data_subset dev

echo 'Compute Features for eval set'

python3 python_scripts/compute_spectrogram_features.py  \
--audio_folder /nas/home/cborrelli/bot_speech/dataset/LA/ASVspoof2019_LA_eval/flac \
--txt_path /nas/home/cborrelli/bot_speech/dataset/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt \
--dest_folder features/spectrogram \
--nfft 512 --hop_size 256 --data_subset eval

python3 python_scripts/compute_spectrogram_features.py  \
--audio_folder /nas/home/cborrelli/bot_speech/dataset/LA/ASVspoof2019_LA_eval/flac \
--txt_path /nas/home/cborrelli/bot_speech/dataset/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt \
--dest_folder features/spectrogram \
--nfft 256 --hop_size 128 --data_subset eval

python3 python_scripts/compute_spectrogram_features.py  \
--audio_folder /nas/home/cborrelli/bot_speech/dataset/LA/ASVspoof2019_LA_eval/flac \
--txt_path /nas/home/cborrelli/bot_speech/dataset/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt \
--dest_folder features/spectrogram \
--nfft 128 --hop_size 64 --data_subset eval

python3 python_scripts/compute_spectrogram_features.py  \
--audio_folder /nas/home/cborrelli/bot_speech/dataset/LA/ASVspoof2019_LA_eval/flac \
--txt_path /nas/home/cborrelli/bot_speech/dataset/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt \
--dest_folder features/spectrogram \
--nfft 64 --hop_size 32 --data_subset eval

