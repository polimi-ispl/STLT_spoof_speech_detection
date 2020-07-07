import pandas as pd
import os
import soundfile as sf
import argparse
import librosa
import numpy as np
import scipy as sp
import scipy.signal
import scipy.stats
import webrtcvad
import struct
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial
import skimage

#import warnings
#warnings.filterwarnings('error')


def float2pcm(sig, dtype='int16'):
    sig = np.asarray(sig)
    if sig.dtype.kind != 'f':
        raise TypeError("'sig' must be a float array")
    dtype = np.dtype(dtype)
    if dtype.kind not in 'iu':
        raise TypeError("'dtype' must be an integer type")

    i = np.iinfo(dtype)
    abs_max = 2 ** (i.bits - 1)
    offset = i.min + abs_max
    return (sig * abs_max + offset).clip(i.min, i.max).astype(dtype)


def run_vad(data, aggress=2, window_duration=0.03, samplerate=16000):
    vad = webrtcvad.Vad()
    vad.set_mode(aggress)
    audio = float2pcm(data)
    raw_samples = struct.pack("%dh" % len(audio), *audio)
    samples_per_window = int(window_duration * samplerate)
    number_windows = int(np.floor(len(audio) / samples_per_window))
    bytes_per_sample = 2

    segments = []
    for i in np.arange(number_windows):
        raw_frame = raw_samples[i * bytes_per_sample * samples_per_window:
                                (i + 1) * bytes_per_sample * samples_per_window]
        is_speech = vad.is_speech(raw_frame, sample_rate=samplerate)
        segments.append(dict(
            start=i * samples_per_window,
            stop=(i + 1) * samples_per_window - 1,
            is_speech=is_speech))

    old_bool = segments[0]['is_speech']
    new_start = segments[0]['start']

    long_segments = []
    for i, segment in enumerate(segments):
        new_bool = segment['is_speech']
        if old_bool == new_bool:
            new_stop = segment['stop']
        else:
            long_segments.append(dict(
                start = new_start,
                stop = new_stop,
                is_speech=old_bool))
            new_start = segment['start']
            new_stop = segment['stop']
        old_bool = new_bool
        if i == len(segments) - 1:
            long_segments.append(dict(
                start = new_start,
                stop = new_stop,
                is_speech = old_bool))
    return long_segments


def compute_lpc(audio, len_lpc, window_length_sec=0.025, samplerate=16000):
    len_lpc = int(len_lpc)
    window_length = int(window_length_sec * samplerate)
    win_number = int(np.floor(len(audio) / window_length))

    coeff = np.zeros((win_number, len_lpc + 1))
    frames = np.zeros((win_number, window_length - 2 * len_lpc - 2))
    res = np.zeros((win_number, window_length - 2 * len_lpc - 2))
    count = 0
    for i in np.arange(win_number):
        frame = audio[i * window_length:(i + 1) * window_length]

        try:
            a = librosa.lpc(frame, len_lpc)
        except:
            count = count + 1
            coeff[i, :] = np.nan
            res[i, :] = np.nan
            frames[i, :] = np.nan
            continue

        coeff[i, :] = a
        frames[i, :] = frame[len_lpc + 1:-len_lpc - 1]
        res_frame = np.asarray(scipy.signal.lfilter(a, [1], frame))
        res[i, :] = res_frame[len_lpc + 1:-len_lpc - 1]

    if count != 0:
        print('{} not valid frames from LPC'.format(count))
    #res = res[~np.isnan(res).any(axis=1)]
    #coeff = coeff[~np.isnan(coeff).any(axis=1)]
    # here I want to remove the samples where res is not valid (for computing gain I need them to be synchronized)
    #frames = frames[~np.isnan(frames).any(axis=1)]

    return res, coeff, frames


def compute_ltp(err, samplerate=16000, analysis_frame_length_sec=0.005):
    analysis_frame_length = int(np.round(analysis_frame_length_sec * samplerate))
    win_number = int(np.floor(len(err) / analysis_frame_length))
    p_min = int(np.round(0.00625 * samplerate))
    p_max = int(np.round(0.025 * samplerate))

    best_res = []
    lpc_err = []

    count = 0
    for i in np.arange(win_number):

        current_start_index = int(i * analysis_frame_length)

        if current_start_index < p_max:
            continue

        if np.any(np.isnan(err[current_start_index - p_max: current_start_index + analysis_frame_length])):
            count = count + 1
            continue

        shifted_frame = err[current_start_index - p_max: current_start_index - p_min + analysis_frame_length]
        current_frame = err[current_start_index: current_start_index + analysis_frame_length]

        b_num = np.correlate(shifted_frame, current_frame, mode='valid')
        b_den = np.correlate(shifted_frame ** 2, np.ones((analysis_frame_length,)), mode='valid')

        b = np.divide(b_num, b_den + np.finfo(float).eps)

        current_frame_rep = np.tile(current_frame, (p_max - p_min + 1, 1))
        scaled_shifted_frame = skimage.util.view_as_windows(shifted_frame, analysis_frame_length,
                                                            step=1) * np.expand_dims(b, axis=1)
        J = np.sum((current_frame_rep - scaled_shifted_frame) ** 2, axis=1)
        m_opt = np.argmin(J)

        best_res.append(current_frame - scaled_shifted_frame[m_opt, :])
        lpc_err.append(current_frame)
    if count != 0:
        print('In this audio {} not valid frames of 0.005 sec were found'.format(count))
    return np.asarray(best_res), np.asarray(lpc_err)


def compute_gain_mean(res, frame):
    gain_frame = np.nanmean(frame ** 2, axis=1) / np.nanmean(res ** 2, axis=1)
    gain = np.nanmean(gain_frame)
    return gain


def compute_gain_max(res, frame):
    gain_frame = np.nanmean(frame ** 2, axis=1) / np.nanmean(res ** 2, axis=1)
    gain_max = np.nanmax(gain_frame)
    return gain_max


def compute_gain_min(res, frame):
    gain_frame = np.nanmean(frame ** 2, axis=1) / np.nanmean(res ** 2, axis=1)
    gain_min= np.nanmin(gain_frame)
    return gain_min


def compute_gain_var(res, frame):
    gain_frame = np.nanmean(frame ** 2, axis=1) / np.nanmean(res ** 2, axis=1)
    gain_var = np.nanvar(gain_frame)
    return gain_var


def compute_res_mean(res):
    res_energy_frame = np.nanmean(res ** 2, axis=1)
    res_energy = np.nanmean(res_energy_frame)
    return res_energy


def compute_res_max(res):
    max_frame = np.nanmax(res, axis=1)
    max_res = np.nanmean(max_frame)
    return max_res


def compute_res_min(res):
    min_frame = np.nanmin(res, axis=1)
    min_res = np.nanmean(min_frame)
    return min_res


def compute_res_var(res):
    var_frame = np.nanvar(res, axis=1)
    var_res = np.nanmean(var_frame)
    return var_res


def compute_features_single(arg, audio_folder, min_length_sec, lpc_length):
    index = arg[0]
    row = arg[1]

    audio_filename = row["audio_filename"] + '.flac'
    data, samplerate = sf.read(os.path.join(audio_folder, audio_filename))

    segments = run_vad(data)

    # Check if at least one voiced segment is present
    voiced_segments = [d for d in segments if
                       d['is_speech'] is True and d['stop'] - d['start'] >= min_length_sec * samplerate]

    # Init data
    data_dict = {'index': index, 'start_voice': None, 'end_voice': None, 'lpc_res_mean': None, 'lpc_res_max': None,
                 'lpc_res_min': None, 'lpc_res_var': None, 'lpc_gain_mean': None, 'lpc_gain_max': None,
                 'lpc_gain_min': None, 'lpc_gain_var': None, 'ltp_res_mean': None, 'ltp_res_max': None,
                 'ltp_res_min': None, 'ltp_res_var': None, 'ltp_gain_mean': None, 'ltp_gain_max': None,
                 'ltp_gain_min': None, 'ltp_gain_var': None}

    if voiced_segments == []:
        return data_dict
    else:
        #  Segment
        voiced = data[voiced_segments[0]['start']: voiced_segments[0]['stop']]

        # Do LPC

        # Store data
        data_dict['index'] = index
        data_dict['start_voice'] = voiced_segments[0]['start']
        data_dict['end_voice'] = voiced_segments[0]['stop']

        res, coeff, frames = compute_lpc(voiced, len_lpc=lpc_length)

        ltp_res, ltp_frames = compute_ltp(res.ravel())

        lpc_res_mean = compute_res_mean(res)
        lpc_res_max = compute_res_max(res)
        lpc_res_min = compute_res_min(res)
        lpc_res_var = compute_res_var(res)

        lpc_gain_mean = compute_gain_mean(res, frames)
        lpc_gain_max = compute_gain_max(res, frames)
        lpc_gain_min = compute_gain_min(res, frames)
        lpc_gain_var = compute_gain_var(res, frames)

        ltp_res_mean = compute_res_mean(ltp_res)
        ltp_res_max = compute_res_max(ltp_res)
        ltp_res_min = compute_res_min(ltp_res)
        ltp_res_var = compute_res_var(ltp_res)

        ltp_gain_mean = compute_gain_mean(ltp_res, ltp_frames)
        ltp_gain_max = compute_gain_max(ltp_res, ltp_frames)
        ltp_gain_min = compute_gain_min(ltp_res, ltp_frames)
        ltp_gain_var = compute_gain_var(ltp_res, ltp_frames)

        # Store data
        data_dict['start_voice'] = voiced_segments[0]['start']
        data_dict['end_voice'] = voiced_segments[0]['stop']

        data_dict['lpc_res_mean'] = lpc_res_mean
        data_dict['lpc_res_max'] = lpc_res_max
        data_dict['lpc_res_min'] = lpc_res_min
        data_dict['lpc_res_var'] = lpc_res_var

        data_dict['lpc_gain_mean'] = lpc_gain_mean
        data_dict['lpc_gain_max'] = lpc_gain_max
        data_dict['lpc_gain_min'] = lpc_gain_min
        data_dict['lpc_gain_var'] = lpc_gain_var

        data_dict['ltp_res_mean'] = ltp_res_mean
        data_dict['ltp_res_max'] = ltp_res_max
        data_dict['ltp_res_min'] = ltp_res_min
        data_dict['ltp_res_var'] = ltp_res_var

        data_dict['ltp_gain_mean'] = ltp_gain_mean
        data_dict['ltp_gain_max'] = ltp_gain_max
        data_dict['ltp_gain_min'] = ltp_gain_min
        data_dict['ltp_gain_var'] = ltp_gain_var


    return data_dict



def compute_features(audio_folder, txt_path, dest_folder, lpc_length, data_subset):
    # Init
    print("Taking files from ", audio_folder, " and doing LPC with length ", str(lpc_length))
    dest_filename = data_subset + '_LPC_' + str(lpc_length) + '.pkl'

    # Check if exists already
    if os.path.exists(os.path.join(dest_folder, dest_filename)):
        print("Already computed!")
        return

    # Open dataset df
    df = pd.read_csv(txt_path, sep=" ", header=None)
    df.columns = ["speaker_id", "audio_filename", "null", "system_id", "label"]
    df = df.drop(columns="null")

    # Paramters
    min_length_sec = 1

    # Prepare parallel execution
    args_list = list(df.iterrows())
    compute_features_partial = partial(compute_features_single, audio_folder=audio_folder,
                                       min_length_sec=min_length_sec, lpc_length=lpc_length)
    # Run parallel execution
    pool = Pool(cpu_count() // 2)
    data_dict_list = list(tqdm(pool.imap(compute_features_partial, args_list), total=len(args_list)))

    #  Convert data to df
    data_list_dict = {k: [dic[k] for dic in data_dict_list] for k in data_dict_list[0]}
    feat = pd.DataFrame.from_dict(data_list_dict)
    feat = feat.set_index(keys='index')
    feat = feat.join(df, how='right')

    #  Save results
    feat.to_pickle(os.path.join(dest_folder, dest_filename))


if __name__ == '__main__':
    # parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_folder', type=str, required=True)
    parser.add_argument('--txt_path', type=str, required=True)
    parser.add_argument('--dest_folder', type=str, required=True)
    parser.add_argument('--lpc_length', type=int, required=True)
    parser.add_argument('--data_subset', type=str, required=True)

    args = parser.parse_args()
    audio_folder = args.audio_folder
    txt_path = args.txt_path
    dest_folder = args.dest_folder
    lpc_length = args.lpc_length
    data_subset = args.data_subset

    compute_features(audio_folder, txt_path, dest_folder, lpc_length, data_subset)




