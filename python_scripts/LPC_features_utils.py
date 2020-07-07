import pandas as pd
import numpy as np
import sklearn.preprocessing
import os
from LPC_params import *



def load_LPC_features(dataset, max_lpc_order = 50, feature_root=feature_root):
    """
    Load train, dev or eval features from feature folder specified in params
    :param dataset:  can be "train", "dev" or "eval"
    :param max_lpc_order: maximum order used in the LPC analysis
    :returns: feature matrix and the labels array
    """
    X_bonafide_list = []
    X_spoof_list = []
    lpc_lengths = np.arange(1, max_lpc_order + 1)

    for l in lpc_lengths:
        features = pd.read_pickle(feature_root + dataset +'_LPC_' + str(l) + '.pkl')
        features.dropna(inplace=True)

        bonafide_features = features[features['label'] == 'bonafide']
        spoof_features = features[features['label'] == 'spoof']

        if l == 1:
            spoof_features = features[features['label'] == 'spoof'].sample(
                n=bonafide_features.shape[0])
            selected_files = spoof_features['audio_filename']
        else:
            spoof_features = spoof_features[spoof_features['audio_filename'].isin(selected_files)]

        X_bonafide_list.append(np.array(bonafide_features['lpc_res_mean']).reshape((-1, 1)))
        X_bonafide_list.append(np.array(bonafide_features['lpc_res_max']).reshape((-1, 1)))
        X_bonafide_list.append(np.array(bonafide_features['lpc_res_min']).reshape((-1, 1)))
        X_bonafide_list.append(np.array(bonafide_features['lpc_res_var']).reshape((-1, 1)))

        X_bonafide_list.append(np.array(bonafide_features['lpc_gain_max']).reshape((-1, 1)))
        X_bonafide_list.append(np.array(bonafide_features['lpc_gain_min']).reshape((-1, 1)))
        X_bonafide_list.append(np.array(bonafide_features['lpc_gain_mean']).reshape((-1, 1)))
        X_bonafide_list.append(np.array(bonafide_features['lpc_gain_var']).reshape((-1, 1)))

        X_bonafide_list.append(np.array(bonafide_features['ltp_res_mean']).reshape((-1, 1)))
        X_bonafide_list.append(np.array(bonafide_features['ltp_res_max']).reshape((-1, 1)))
        X_bonafide_list.append(np.array(bonafide_features['ltp_res_min']).reshape((-1, 1)))
        X_bonafide_list.append(np.array(bonafide_features['ltp_res_var']).reshape((-1, 1)))

        X_bonafide_list.append(np.array(bonafide_features['ltp_gain_max']).reshape((-1, 1)))
        X_bonafide_list.append(np.array(bonafide_features['ltp_gain_min']).reshape((-1, 1)))
        X_bonafide_list.append(np.array(bonafide_features['ltp_gain_mean']).reshape((-1, 1)))
        X_bonafide_list.append(np.array(bonafide_features['ltp_gain_var']).reshape((-1, 1)))

        X_spoof_list.append(np.array(spoof_features['lpc_res_mean']).reshape((-1, 1)))
        X_spoof_list.append(np.array(spoof_features['lpc_res_max']).reshape((-1, 1)))
        X_spoof_list.append(np.array(spoof_features['lpc_res_min']).reshape((-1, 1)))
        X_spoof_list.append(np.array(spoof_features['lpc_res_var']).reshape((-1, 1)))

        X_spoof_list.append(np.array(spoof_features['lpc_gain_max']).reshape((-1, 1)))
        X_spoof_list.append(np.array(spoof_features['lpc_gain_min']).reshape((-1, 1)))
        X_spoof_list.append(np.array(spoof_features['lpc_gain_mean']).reshape((-1, 1)))
        X_spoof_list.append(np.array(spoof_features['lpc_gain_var']).reshape((-1, 1)))

        X_spoof_list.append(np.array(spoof_features['ltp_res_mean']).reshape((-1, 1)))
        X_spoof_list.append(np.array(spoof_features['ltp_res_max']).reshape((-1, 1)))
        X_spoof_list.append(np.array(spoof_features['ltp_res_min']).reshape((-1, 1)))
        X_spoof_list.append(np.array(spoof_features['ltp_res_var']).reshape((-1, 1)))

        X_spoof_list.append(np.array(spoof_features['ltp_gain_max']).reshape((-1, 1)))
        X_spoof_list.append(np.array(spoof_features['ltp_gain_min']).reshape((-1, 1)))
        X_spoof_list.append(np.array(spoof_features['ltp_gain_mean']).reshape((-1, 1)))
        X_spoof_list.append(np.array(spoof_features['ltp_gain_var']).reshape((-1, 1)))

    X_bonafide = np.concatenate(X_bonafide_list, axis=1)
    X_spoof = np.concatenate(X_spoof_list, axis=1)
    X = np.concatenate((X_bonafide, X_spoof), axis=0)

    y_bonafide = np.ones((X_bonafide.shape[0],))
    y_spoof = np.zeros((X_spoof.shape[0],))

    y = np.concatenate((y_bonafide, y_spoof), axis=0)

    return X, y

#TODO: define a smarter feature loader for having the same number of samples for each class


def reshape_LPC_features(X, n_features=16, lpc_length=50 ):
    """
    Reshape features for pre processing to the CNN
    :param X: feature matrix of dimension N_samples x (LPC_length*n_features)
    :param n_features: int, number of considered features
    :param lpc_length: int, maximum lenght of the LPC analysis
    :return: feature matrix of dimension N_samples x n_features x LPC_length
    """

    reshaped_X = np.zeros([X.shape[0], n_features, lpc_length])

    for n in np.arange(X.shape[0]):
        for i in np.arange(n_features):
            reshaped_X[n, i, :] = X[n, i::n_features]

    return reshaped_X


def preprocess_LPC_features(X, p_axis=(0), **kwargs):
    """
    Normalize feature following one algorithm between

    """
    n_features = X.shape[1]

    norm_X = np.zeros(X.shape)


    if 'minmax_scaler' in kwargs.keys():
        feat_min, feat_max = kwargs['minmax_scaler']

        if not np.any(feat_min) and not np.any(feat_max):
            feat_min = np.expand_dims(X.min(axis=p_axis), axis=p_axis)
            feat_max = np.expand_dims(X.max(axis=p_axis), axis=p_axis)

        norm_X = (X - feat_min) / (feat_max - feat_min)
        return_args = [norm_X, {'minmax_scaler': [feat_min, feat_max]}]

    elif 'z_score' in kwargs.keys():
        feat_mu, feat_std = kwargs['z_score']

        if not np.any(feat_mu) and not np.any(feat_std):
            feat_mu = np.expand_dims(X.mean(axis=p_axis), axis=p_axis)
            feat_std = np.expand_dims(X.std(axis=p_axis), axis=p_axis)

        norm_X = (X - feat_mu) / feat_std
        return_args = [norm_X, {'z_score': [feat_mu, feat_std]}]

    elif 'l2_norm' in kwargs.keys():
        for feature_index in np.arange(n_features):
            norm_X[:, feature_index, :] = sklearn.preprocessing.normalize(X[:, feature_index, :], norm='l2')
        return_args = [norm_X, {'l2_norm': [None, None]}]

    elif 'l1_norm' in kwargs.keys():
        for feature_index in np.arange(n_features):
            norm_X[:, feature_index, :] = sklearn.preprocessing.normalize(X[:, feature_index, :], norm='l1')
        return_args = [norm_X, {'l1_norm': [None, None]}]

    elif 'max_norm' in kwargs.keys():
        for feature_index in np.arange(n_features):
            norm_X[:, feature_index, :] = sklearn.preprocessing.normalize(X[:, feature_index, :], norm='max')
        return_args = [norm_X, {'max_norm': [None, None]}]

    elif 'no_norm' in kwargs.keys():
        norm_X = X
        return_args = [norm_X, {'no_norm': [None, None]}]

    else:
        raise NotImplementedError('Normalization not defined')

    return return_args