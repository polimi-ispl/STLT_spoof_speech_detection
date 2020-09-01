import pandas as pd
import numpy as np
import sys
import os
import scipy.stats as stats
import time
import tqdm
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import *
from sklearn.svm import *
from sklearn.pipeline import *
from sklearn.model_selection import *
import itertools
import argparse
from sklearn.preprocessing import normalize
from sklearn.ensemble import RandomForestClassifier

unknown_label = 7
unknown_number = 2

def_selected_features = ['lpc', 'bicoh', 'unet']
def_number_lpc_order = 49
def_stop_lpc_order = 50

multiclass_list = ['-', 'A01', 'A02', 'A03', 'A04', 'A05', 'A06']

normalization_step = {"minmax_norm": MinMaxScaler(), "zscore_norm": StandardScaler(), "l2_norm": Normalizer()}
classification_step = {"svm": SVC(random_state=2), "rf": RandomForestClassifier(random_state=2)}


def load_features(selected_features, number_lpc_order, stop_lpc_order, nfft, hop_size):
    bicoh_train_feat_path = '../features/bicoherences/dataframes/train_bicoh_stats_nfft_{}_hop_size_{}.pkl'.format(
        nfft, hop_size)
    lpc_train_feat_path = '../features/lpc/dataframe/train.pkl'
    unet_train_feat_path = '../features/unet/train_nfft_{}_hop_size_{}.pkl'.format(nfft, hop_size)

    bicoh_dev_feat_path = '../features/bicoherences/dataframes/dev_bicoh_stats_nfft_{}_hop_size_{}.pkl'.format(
        nfft, hop_size)
    lpc_dev_feat_path = '../features/lpc/dataframe/dev.pkl'
    unet_dev_feat_path = '../features/unet/dev_nfft_{}_hop_size_{}.pkl'.format(nfft, hop_size)

    bicoh_eval_feat_path = '../features/bicoherences/dataframes/eval_bicoh_stats_nfft_{}_hop_size_{}.pkl'.format(
        nfft, hop_size)
    lpc_eval_feat_path = '../features/lpc/dataframe/eval.pkl'
    unet_eval_feat_path = '../features/unet/eval_nfft_{}_hop_size_{}.pkl'.format(nfft, hop_size)

    lpc_linspace = np.linspace(start=stop_lpc_order - number_lpc_order, stop=stop_lpc_order, dtype=int)

    lpc_selected_orders = []
    for l in lpc_linspace:
        lpc_selected_orders.append(str(l))

    lpc_selected_orders = tuple(lpc_selected_orders)

    train_features = pd.DataFrame()
    dev_features = pd.DataFrame()
    eval_features = pd.DataFrame()

    for feat in selected_features:
        if feat == 'lpc':
            lpc_feat_train = pd.read_pickle(lpc_train_feat_path)
            lpc_feat_train.set_index('audio_filename', inplace=True)

            lpc_feat_dev = pd.read_pickle(lpc_dev_feat_path)
            lpc_feat_dev.set_index('audio_filename', inplace=True)

            lpc_feat_eval = pd.read_pickle(lpc_eval_feat_path)
            lpc_feat_eval.set_index('audio_filename', inplace=True)

            drop_list = [a for a in lpc_feat_train.columns if a.startswith(('lpc', 'ltp'))
                         and not a.endswith(lpc_selected_orders)]

            lpc_feat_train = lpc_feat_train.drop(drop_list, axis=1)
            lpc_feat_dev = lpc_feat_dev.drop(drop_list, axis=1)
            lpc_feat_eval = lpc_feat_eval.drop(drop_list, axis=1)

            if train_features.empty:
                train_features = lpc_feat_train.copy()

                dev_features = lpc_feat_dev.copy()

                eval_features = lpc_feat_eval.copy()
            else:
                train_features = pd.concat([train_features, lpc_feat_train], axis=1)
                dev_features = pd.concat([dev_features, lpc_feat_dev], axis=1)
                eval_features = pd.concat([eval_features, lpc_feat_eval], axis=1)

        elif feat == 'bicoh':
            bicoh_feat_train = pd.read_pickle(bicoh_train_feat_path)
            bicoh_feat_train.set_index('audio_filename', inplace=True)

            bicoh_feat_dev = pd.read_pickle(bicoh_dev_feat_path)
            bicoh_feat_dev.set_index('audio_filename', inplace=True)

            bicoh_feat_eval = pd.read_pickle(bicoh_eval_feat_path)
            bicoh_feat_eval.set_index('audio_filename', inplace=True)

            if train_features.empty:
                train_features = bicoh_feat_train.copy()

                dev_features = bicoh_feat_dev.copy()

                eval_features = bicoh_feat_eval.copy()
            else:
                train_features = pd.concat([train_features, bicoh_feat_train], axis=1)
                dev_features = pd.concat([dev_features, bicoh_feat_dev], axis=1)
                eval_features = pd.concat([eval_features, bicoh_feat_eval], axis=1)

        elif feat == 'unet':
            unet_feat_train = pd.read_pickle(unet_train_feat_path)
            unet_feat_train.set_index('audio_filename', inplace=True)

            unet_feat_dev = pd.read_pickle(unet_dev_feat_path)
            unet_feat_dev.set_index('audio_filename', inplace=True)

            unet_feat_eval = pd.read_pickle(unet_eval_feat_path)
            unet_feat_eval.set_index('audio_filename', inplace=True)

            if train_features.empty:
                train_features = unet_feat_train.copy()

                dev_features = unet_feat_dev.copy()

                eval_features = unet_feat_eval.copy()
            else:
                train_features = pd.concat([train_features, unet_feat_train], axis=1)
                dev_features = pd.concat([dev_features, unet_feat_dev], axis=1)
                eval_features = pd.concat([eval_features, unet_feat_eval], axis=1)

    # remove duplicates from dataframes
    train_features = train_features.loc[:, ~train_features.columns.duplicated()]
    dev_features = dev_features.loc[:, ~dev_features.columns.duplicated()]
    eval_features = eval_features.loc[:, ~eval_features.columns.duplicated()]

    # reset index after aggregating by audio filename
    train_features.reset_index(inplace=True)
    dev_features.reset_index(inplace=True)
    eval_features.reset_index(inplace=True)

    # drop irrelevant columns for classification
    train_features.drop(['audio_filename', 'end_voice', 'start_voice', 'speaker_id', 'label'], axis=1, inplace=True)
    dev_features.drop(['audio_filename', 'end_voice', 'start_voice', 'speaker_id', 'label'], axis=1, inplace=True)
    eval_features.drop(['audio_filename', 'end_voice', 'start_voice', 'speaker_id', 'label'], axis=1, inplace=True)

    return train_features, dev_features, eval_features



def train_one_configuration(n_key, c_key, u, df_train, df_dev, df_eval):
    multiclass_dict = {'-': 0, 'A01': 1, 'A02': 2, 'A03': 3, 'A04': 4, 'A05': 5, 'A06': 6}

    if u[0] == '-' or u[1] == '-':
        return

    # label 8 corresponds to unknown known
    for i in range(len(u)):
        multiclass_dict[u[i]] = unknown_label

    X_train = df_train.loc[:, df_train.columns != 'system_id'].values
    X_dev = df_dev.loc[:, df_dev.columns != 'system_id'].values
    X_eval = df_eval.loc[:, df_eval.columns != 'system_id'].values

    y_train_open_set = df_train.loc[:, 'system_id'].values
    y_train_open_set = [multiclass_dict[a] for a in y_train_open_set]

    y_dev_open_set = df_dev.loc[:, 'system_id'].values
    y_dev_open_set = [multiclass_dict[a] for a in y_dev_open_set]

    y_eval_open_set = unknown_label * np.ones(X_eval.shape[0])

    X = X_train
    y = y_train_open_set


    # Define the pipeline
    steps = [('norm', normalization_step[n_key]), ('class', classification_step[c_key])]
    pipeline = Pipeline(steps)

    if c_key == 'svm':
        param_grid = {'class__C': [0.1, 1, 10, 100, 1000],
                      'class__gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                      'class__kernel': ['rbf', 'linear', 'sigmoid']
                      }
    elif c_key == 'rf':
        param_grid = {'class__n_estimators': [100, 300, 500, 800, 1200],
                      'class__max_depth': [5, 8, 15, 25, 30],
                      'class__min_samples_split': [2, 5, 10, 15, 100],
                      'class__min_samples_leaf': [1, 2, 5, 10]
                      }


    search = GridSearchCV(pipeline, param_grid=param_grid,  n_jobs=1)
    search.fit(X, y)
    model = search.best_estimator_

    model.fit(X, y)

    y_predict_dev = model.predict(X_dev)
    y_predict_eval = model.predict(X_eval)
    y_predict_train = model.predict(X_train)

    cm_train = confusion_matrix(y_train_open_set, y_predict_train, normalize='true')
    cm_dev = confusion_matrix(y_dev_open_set, y_predict_dev, normalize='true')
    cm_eval = confusion_matrix(y_eval_open_set, y_predict_eval, normalize='true')

    return cm_train, cm_dev, cm_eval


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nfft', type=int, required=True)
    parser.add_argument('--hop_size', type=int, required=True)
    parser.add_argument('--selected_features', nargs='+', required=False, default=def_selected_features)
    parser.add_argument('--stop_lpc_order', type=int, required=False, default=def_stop_lpc_order)
    parser.add_argument('--number_lpc_order', type=int, required=False, default=def_number_lpc_order)

    args = parser.parse_args()
    nfft = args.nfft
    hop_size = args.hop_size
    selected_features = args.selected_features
    stop_lpc_order = args.stop_lpc_order
    number_lpc_order = args.number_lpc_order

    unknown_combinations = itertools.combinations(multiclass_list, unknown_number)

    # TODO: call train_one_configuration for different elements of unknown combinations and different normalization +
    #  classification steps
    n_keys = list(normalization_step.keys())
    c_keys = list(classification_step.keys())
    uu = list(unknown_combinations)
    nfft = 128
    hop_size = 64
    #print(n_keys[0])
    # Load features
    df_train, df_dev, df_eval = load_features(selected_features=selected_features,
                                              number_lpc_order=number_lpc_order,
                                              stop_lpc_order=stop_lpc_order,
                                              nfft=nfft,
                                              hop_size=hop_size)
    # Prepare data for open set
    train_one_configuration(u=uu[0], n_key=n_keys[0], c_key=c_keys[0], df_train=df_train, df_dev=df_dev, df_eval=df_eval)
    #
    #
    #
    #
    #
    # ]
    # for u in unknown_combinations:
    #     for f in fft_params:
    #         for p in pipeline_list:
    #             train_one_configuration(p, u, f[0], f[1])
