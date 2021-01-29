import pandas as pd
import numpy as np
import os
import time
import pickle
from sklearn.preprocessing import *
from sklearn.svm import *
from sklearn.pipeline import *
from sklearn.metrics import accuracy_score
from sklearn.model_selection import *
import itertools
import argparse
from sklearn.ensemble import RandomForestClassifier
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')

feature_root_path = '/nas/home/cborrelli/bot_speech/features'

def load_features():
    logging.debug("Loading features")
    lpc_train_feat_path = os.path.join(feature_root_path, 'lpc/dataframe/train.pkl')
    lpc_dev_feat_path = os.path.join(feature_root_path, 'lpc/dataframe/dev.pkl')
    lpc_eval_feat_path = os.path.join(feature_root_path, 'lpc/dataframe/eval.pkl')

    lpc_feat_train = pd.read_pickle(lpc_train_feat_path)
    lpc_feat_train.set_index('audio_filename', inplace=True)

    lpc_feat_dev = pd.read_pickle(lpc_dev_feat_path)
    lpc_feat_dev.set_index('audio_filename', inplace=True)

    lpc_feat_eval = pd.read_pickle(lpc_eval_feat_path)
    lpc_feat_eval.set_index('audio_filename', inplace=True)

    train_features = lpc_feat_train.copy()
    dev_features = lpc_feat_dev.copy()
    eval_features = lpc_feat_eval.copy()

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
    train_features.drop(['audio_filename', 'end_voice', 'start_voice', 'speaker_id', 'system_id'], axis=1, inplace=True)
    dev_features.drop(['audio_filename', 'end_voice', 'start_voice', 'speaker_id', 'system_id'], axis=1, inplace=True)
    eval_features.drop(['audio_filename', 'end_voice', 'start_voice', 'speaker_id', 'system_id'], axis=1, inplace=True)

    return train_features, dev_features, eval_features


def train_single_order(df_train, df_dev, df_eval, result_filename):
    logging.debug("Training order by order")

    # Collect labels
    binary_dict = {'bonafide': 0, 'spoof': 1}

    y_train = df_train.loc[:, 'label'].values
    y_train = np.array([binary_dict[a] for a in y_train])

    y_dev = df_dev.loc[:, 'label'].values
    y_dev = np.array([binary_dict[a] for a in y_dev])

    y_eval = df_eval.loc[:, 'label'].values
    y_eval = np.array([binary_dict[a] for a in y_eval])

    # Define pipeline
    steps = [('norm', MinMaxScaler()),
             ('class', SVC(random_state=2, class_weight='balanced', C=1000,
                           gamma='auto', kernel='rbf', decision_function_shape='ovo'))]
    model = Pipeline(steps)

    # Selected LPC orders
    stop_lpc_order = 50
    number_lpc_order = 49

    lpc_linspace = np.arange(start=stop_lpc_order - number_lpc_order, stop=stop_lpc_order + 1, dtype=int)

    results_single_order = []
    for l in lpc_linspace:
        logging.debug('Training for order {}'.format(l))
        start_time = time.time()

        order = '_' + str(l)
        keep_list = [a for a in df_train.columns if a.startswith(('lpc', 'ltp'))
                     and a.endswith(order)]

        X_train = df_train.loc[:, keep_list].values
        X_dev = df_dev.loc[:, keep_list].values
        X_eval = df_eval.loc[:, keep_list].values

        model.fit(X_train, y_train)

        y_predict_dev = model.predict(X_dev)
        y_predict_eval = model.predict(X_eval)
        y_predict_train = model.predict(X_train)

        results = {
            #'y_train': y_train,
            #'y_predict_train': y_predict_train,
            #'y_dev': y_dev,
            #'y_predict_dev': y_predict_dev,
            #'y_eval': y_eval,
            #'y_predict_eval': y_predict_eval,
            'lpc_order':l,
            'train_accuracy': accuracy_score(y_train, y_predict_train),
            'dev_accuracy': accuracy_score(y_dev, y_predict_dev),
            'eval_accuracy': accuracy_score(y_eval, y_predict_eval)
        }
        results_single_order.append(results)
        logging.debug("Training time : {}".format(str(time.time() - start_time)))

    # Save dictionary
    with open(result_filename, 'wb') as handle:
        pickle.dump(results_single_order, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Return as dataframe
    df_results_single_order = pd.DataFrame.from_dict(results_single_order)
    return df_results_single_order


def training_multiple_orders(df_results_single_order, df_train, df_dev, df_eval):
    logging.debug("Training for increasing order")
    # Order single order results by eval accuracy
    df_results_single_order.sort_values(by='eval_accuracy')
    ordered_lpc_orders = df_results_single_order['lpc']

    # Collect labels
    binary_dict = {'bonafide': 0, 'spoof': 1}

    y_train = df_train.loc[:, 'label'].values
    y_train = np.array([binary_dict[a] for a in y_train])

    y_dev = df_dev.loc[:, 'label'].values
    y_dev = np.array([binary_dict[a] for a in y_dev])

    y_eval = df_eval.loc[:, 'label'].values
    y_eval = np.array([binary_dict[a] for a in y_eval])

    # Define pipeline
    steps = [('norm', MinMaxScaler()),
             ('class', SVC(random_state=2, class_weight='balanced', C=1000,
                           gamma='auto', kernel='rbf', decision_function_shape='ovo'))]
    model = Pipeline(steps)

    selected_orders = []
    for l in ordered_lpc_orders:
        selected_orders.append(l)

        keep_list = [a for a in df_train.columns if a.startswith(('lpc', 'ltp'))
                     and a.endswith(order)]



if __name__ == '__main__':
    result_root_path = "/nas/home/cborrelli/bot_speech/results/closed_set_binary_lpc_order"
    # Load features
    df_train, df_dev, df_eval = load_features()

    df_results_single_order = train_single_order(df_train=df_train,df_dev=df_dev, df_eval=df_eval,
                                                 result_filename=os.path.join(result_root_path,
                                                                              'results_single_order.npy'))
    training_multiple_orders(df_results_single_order, df_train, df_dev, df_eval)