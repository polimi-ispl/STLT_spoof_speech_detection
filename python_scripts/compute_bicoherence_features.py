from scipy import signal
import numpy as np
import soundfile as sf
import os
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial
import argparse


def compute_bicoherence(arg, dest_root, audio_folder, nperseg=512, noverlap=256):
    """ Compute the bicoherence of a signal
    using the function scipy.signal.spectrogram
    """
    index = arg[0]
    row = arg[1]
    audio_filename = row["audio_filename"] + '.flac'
    audio, rate = sf.read(os.path.join(audio_folder, audio_filename))
    # compute the stft
    f_axis, t_axis, spec = signal.spectrogram(audio, fs=rate, nperseg=nperseg, noverlap=noverlap, mode='complex',
                                              return_onesided=False)

    # transpose (f, t) -> (t, f)
    spec = np.transpose(spec, [1, 0])

    # compute the bicoherence
    arg = np.arange(-f_axis.size / 4, f_axis.size / 4, dtype='int')
    sumarg = arg[:, None] + arg[None, :]

    num = np.mean(spec[:, arg, None] * spec[:, None, arg] *
                  np.conjugate(spec[:, sumarg]),
                  axis=0)

    denum = np.sqrt(np.mean(
        np.abs(spec[:, arg, None] * spec[:, None, arg]) ** 2, axis=0) * np.mean(
        np.abs(np.conjugate(spec[:, sumarg])) ** 2,
        axis=0))

    bicoh = num / denum

    bicoh_out_name = os.path.join(dest_root, row['audio_filename']+'.npy')
    np.save(bicoh_out_name, bicoh)

    return



def compute_features(audio_folder, txt_path, dest_root, data_subset, nfft, hop_size):

    # Open dataset df
    df = pd.read_csv(txt_path, sep=" ", header=None)
    df.columns = ["speaker_id", "audio_filename", "null", "system_id", "label"]
    df = df.drop(columns="null")


    # Prepare parallel execution
    args_list = list(df.iterrows())

    dest_subset_folder = '{}_nfft_{}_hop_size_{}'.format(data_subset, nfft, hop_size)
    dest_subset_root = os.path.join(dest_root, dest_subset_folder)

    if not os.path.exists(dest_subset_root):
        os.mkdir(dest_subset_root)

    compute_features_partial = partial(compute_bicoherence, dest_root=dest_subset_root, audio_folder=audio_folder,
                                       nperseg=nfft, noverlap=hop_size)
    # Run parallel execution
    pool = Pool(cpu_count() // 2)
    _ = list(tqdm(pool.imap(compute_features_partial, args_list), total=len(args_list)))



if __name__ == '__main__':
    # parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_folder', type=str, required=True)
    parser.add_argument('--txt_path', type=str, required=True)
    parser.add_argument('--dest_folder', type=str, required=True)
    parser.add_argument('--nfft', type=int, required=True)
    parser.add_argument('--hop_size', type=int, required=True)
    parser.add_argument('--data_subset', type=str, required=True)

    args = parser.parse_args()
    audio_folder = args.audio_folder
    txt_path = args.txt_path
    dest_folder = args.dest_folder
    nfft = args.nfft
    hop_size = args.hop_size
    data_subset = args.data_subset

    compute_features(audio_folder, txt_path, dest_folder, data_subset, nfft, hop_size)
