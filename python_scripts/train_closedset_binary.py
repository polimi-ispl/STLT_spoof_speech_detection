import pandas as pd
import numpy as np
import os
import time
import pickle
from sklearn.preprocessing import *
from sklearn.svm import *
from sklearn.pipeline import *
from sklearn.model_selection import *
import itertools
import argparse
from sklearn.ensemble import RandomForestClassifier
import logging

os.nice(3)

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')

result_root_path = '/nas/home/cborrelli/bot_speech/results/closed_set_binary'
feature_root_path = '/nas/home/cborrelli/bot_speech/features'

unknown_label = 7
unknown_number = 2

def_nfft = [512]
def_hop_size = [256]
def_selected_features = ['lpc', 'bicoh', 'unet_norm']
def_number_lpc_order = 49
def_stop_lpc_order = 50
def_normalizers_keys = ["minmax", "zscore", "l2"]
def_classifiers_keys = ["svm", "rf"]

normalizers = {"minmax": MinMaxScaler(), "zscore": StandardScaler(), "l2": Normalizer()}
classifiers = {"svm": SVC(random_state=2, class_weight='balanced'), "rf": RandomForestClassifier(random_state=2)}


def load_features(selected_features, number_lpc_order, stop_lpc_order, nfft, hop_size):
    logging.debug("Loading features")
    bicoh_train_feat_path = os.path.join(
        feature_root_path,
        'bicoherences/dataframes/train_bicoh_stats_nfft_{}_hop_size_{}.pkl'.format(nfft, hop_size))
    lpc_train_feat_path = os.path.join(feature_root_path, 'lpc/dataframe/train.pkl')
    unet_train_feat_path = os.path.join(feature_root_path, 'unet/train_nfft_{}_hop_size_{}.pkl'.format(nfft, hop_size))
    unet_norm_train_feat_path = os.path.join(feature_root_path, 'unet_norm/train_nfft_{}_hop_size_{}.pkl'.format(nfft, hop_size))

    bicoh_dev_feat_path = os.path.join(
        feature_root_path,
        'bicoherences/dataframes/dev_bicoh_stats_nfft_{}_hop_size_{}.pkl'.format(nfft, hop_size))
    lpc_dev_feat_path = os.path.join(feature_root_path, 'lpc/dataframe/dev.pkl')
    unet_dev_feat_path = os.path.join(feature_root_path, 'unet/dev_nfft_{}_hop_size_{}.pkl'.format(nfft, hop_size))
    unet_norm_dev_feat_path = os.path.join(feature_root_path, 'unet_norm/dev_nfft_{}_hop_size_{}.pkl'.format(nfft, hop_size))

    bicoh_eval_feat_path = os.path.join(
        feature_root_path,
        'bicoherences/dataframes/eval_bicoh_stats_nfft_{}_hop_size_{}.pkl'.format(
            nfft, hop_size))
    lpc_eval_feat_path = os.path.join(feature_root_path, 'lpc/dataframe/eval.pkl')
    unet_eval_feat_path = os.path.join(feature_root_path, 'unet/eval_nfft_{}_hop_size_{}.pkl'.format(nfft, hop_size))
    unet_norm_eval_feat_path = os.path.join(feature_root_path, 'unet_norm/eval_nfft_{}_hop_size_{}.pkl'.format(nfft, hop_size))

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

        elif feat == 'unet_norm':
            new_norm_feat_columns = ['speaker_id', 'audio_filename', 'system_id', 'label',
                                     'unet_norm_mse_alg_A01', 'unet_norm_mse_alg_A02', 'unet_norm_mse_alg_A03',
                                     'unet_norm_mse_alg_A04', 'unet_norm_mse_alg_A05', 'unet_norm_mse_alg_A06']
            unet_norm_feat_train = pd.read_pickle(unet_norm_train_feat_path)
            unet_norm_feat_train.columns = new_norm_feat_columns
            unet_norm_feat_train.set_index('audio_filename', inplace=True)

            unet_norm_feat_dev = pd.read_pickle(unet_norm_dev_feat_path)
            unet_norm_feat_dev.columns = new_norm_feat_columns
            unet_norm_feat_dev.set_index('audio_filename', inplace=True)

            unet_norm_feat_eval = pd.read_pickle(unet_norm_eval_feat_path)
            unet_norm_feat_eval.columns = new_norm_feat_columns
            unet_norm_feat_eval.set_index('audio_filename', inplace=True)

            if train_features.empty:
                train_features = unet_norm_feat_train.copy()

                dev_features = unet_norm_feat_dev.copy()

                eval_features = unet_norm_feat_eval.copy()
            else:
                train_features = pd.concat([train_features, unet_norm_feat_train], axis=1)
                dev_features = pd.concat([dev_features, unet_norm_feat_dev], axis=1)
                eval_features = pd.concat([eval_features, unet_norm_feat_eval], axis=1)
    # remove NaN from dataframes
    train_features.dropna(inplace=True, axis=0)
    dev_features.dropna(inplace=True, axis=0)
    eval_features.dropna(inplace=True, axis=0)

    # remove duplicates from dataframes
    train_features = train_features.loc[:, ~train_features.columns.duplicated()]
    dev_features = dev_features.loc[:, ~dev_features.columns.duplicated()]
    eval_features = eval_features.loc[:, ~eval_features.columns.duplicated()]

    # reset index after aggregating by audio filename
    train_features.reset_index(inplace=True)
    dev_features.reset_index(inplace=True)
    eval_features.reset_index(inplace=True)

    # drop irrelevant columns for classification
    if "lpc" in selected_features:
        train_features.drop(['audio_filename', 'end_voice', 'start_voice', 'speaker_id', 'system_id'], axis=1, inplace=True)
        dev_features.drop(['audio_filename', 'end_voice', 'start_voice', 'speaker_id', 'system_id'], axis=1, inplace=True)
        eval_features.drop(['audio_filename', 'end_voice', 'start_voice', 'speaker_id', 'system_id'], axis=1, inplace=True)
    else:
        train_features.drop(['audio_filename', 'speaker_id', 'system_id'], axis=1, inplace=True)
        dev_features.drop(['audio_filename', 'speaker_id', 'system_id'], axis=1, inplace=True)
        eval_features.drop(['audio_filename', 'speaker_id', 'system_id'], axis=1, inplace=True)

    return train_features, dev_features, eval_features


def train_one_configuration(n_key, c_key, df_train, df_dev, df_eval, result_filename):
    #if os.path.exists(result_filename):
    #    logging.debug("Results already computed")
    #    return

    binary_dict = {'bonafide': 0, 'spoof': 1}

    X_train = df_train.loc[:, df_train.columns != 'label'].values
    X_dev = df_dev.loc[:, df_dev.columns != 'label'].values
    X_eval = df_eval.loc[:, df_eval.columns != 'label'].values

    y_train = df_train.loc[:, 'label'].values
    y_train = np.array([binary_dict[a] for a in y_train])

    y_dev = df_dev.loc[:, 'label'].values
    y_dev = np.array([binary_dict[a] for a in y_dev])

    y_eval = df_eval.loc[:, 'label'].values
    y_eval = np.array([binary_dict[a] for a in y_eval])

    X, X_grid, y, y_grid = train_test_split(X_train, y_train, test_size=0.25, random_state=2)  # 0.25 x 0.8 = 0.2

    # Define the pipeline
    steps = [('norm', normalizers[n_key]), ('class', classifiers[c_key])]
    pipeline = Pipeline(steps)

    if c_key == 'svm':
        param_grid = [{'class__C': [0.1, 1, 10, 100, 1000],
                       'class__gamma': ['scale', 'auto', 1, 0.1, 0.01],
                       'class__kernel': ['rbf'],
                       'class__decision_function_shape': ['ovo', 'ovr']
                       }, {'class__C': [0.1, 1, 10, 100, 1000],
                           'class__kernel': ['linear'],
                           'class__decision_function_shape': ['ovo', 'ovr']}]
    elif c_key == 'rf':
        param_grid = {'class__n_estimators': [10, 100, 500, 1000],
                      'class__max_depth': [30, None],
                      'class__min_samples_split': [2],
                      'class__min_samples_leaf': [1],
                      'class__criterion': ['gini', 'entropy']
                      }
    else:
        print("Wrong classifier name")
        return

    logging.debug("Grid search")
    search = GridSearchCV(pipeline, param_grid=param_grid, n_jobs=24, verbose=1, cv=2, scoring='balanced_accuracy', return_train_score=True)
    # Perform grid search on dev subset
    search.fit(X_grid, y_grid)
    model = search.best_estimator_
    logging.debug("Fit best model")

    model.fit(X, y)

    y_predict_dev = model.predict(X_dev)
    y_predict_eval = model.predict(X_eval)
    y_predict_train = model.predict(X_train)
    results = {
        'y_train': y_train,
        'y_predict_train': y_predict_train,
        'y_dev': y_dev,
        'y_predict_dev': y_predict_dev,
        'y_eval': y_eval,
        'y_predict_eval': y_predict_eval,
        'best_model': search.best_params_
    }

    logging.debug("Save results")

    with open(result_filename, 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nfft', nargs='+', required=False, default=def_nfft)
    parser.add_argument('--hop_size', nargs='+', required=False, default=def_hop_size)
    parser.add_argument('--selected_features', nargs='+', required=False, default=def_selected_features)
    parser.add_argument('--stop_lpc_order', type=int, required=False, default=def_stop_lpc_order)
    parser.add_argument('--number_lpc_order', type=int, required=False, default=def_number_lpc_order)
    parser.add_argument('--classifiers', nargs='+', required=False, default=def_classifiers_keys)
    parser.add_argument('--normalizers', nargs='+', required=False, default=def_normalizers_keys)

    args = parser.parse_args()
    nfft_list = args.nfft
    hop_size_list = args.hop_size
    selected_features = args.selected_features
    stop_lpc_order = args.stop_lpc_order
    number_lpc_order = args.number_lpc_order
    classifiers_keys = args.classifiers
    normalizers_keys = args.normalizers

    # Load features

    # Prepare data for open set
    for fft_params in zip(nfft_list, hop_size_list):
        nfft = fft_params[0]
        hop_size = fft_params[1]
        df_train, df_dev, df_eval = load_features(selected_features=selected_features,
                                                  number_lpc_order=number_lpc_order,
                                                  stop_lpc_order=stop_lpc_order,
                                                  nfft=nfft,
                                                  hop_size=hop_size)
        for c in classifiers_keys:
            logging.debug("Classifier {}".format(c))
            for n in normalizers_keys:
                logging.debug("Normalization {}".format(n))
                result_name = "class_{}_norm_{}_nfft_{}_hop-size_{}_numberlpcorder_{}_stoplpcorder_{}".format(
                    c, n, nfft, hop_size,
                    number_lpc_order,
                    stop_lpc_order)

                result_name = result_name + "_selected_features_" + "-".join(
                    s for s in selected_features) + ".npy"

                result_filename = os.path.join(result_root_path, result_name)
                train_one_configuration(n_key=n,
                                        c_key=c, df_train=df_train,
                                        df_dev=df_dev, df_eval=df_eval,
                                        result_filename=result_filename)

    logging.debug("Finished")

