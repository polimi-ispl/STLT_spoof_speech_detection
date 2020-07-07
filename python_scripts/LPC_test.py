import argparse
import os
import tensorflow.keras
from LPC_params import *
from LPC_features_utils import *
import cnn_utils
from metrics import *
import numpy as np



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('--train', help='Path of the dataset used for training', type=str, required=True)
    #parser.add_argument('--dev', help='Path of the dataset used for development', type=str, required=True)
    #parser.add_argument('--eval', help='Path of the dataset used for evaluation', type=str, required=True)

    parser.add_argument('--preprocessing', help='Preprocessing algorithm', type=str, required=True, default='l2_norm')
    parser.add_argument('--preprocessing_axis', help='Axis used for preprocessing operation (among all samples or along lpc axis', type=int, nargs='+', required=True, default='1')
    parser.add_argument('--cnn_model', help='CNN model', type=str, required=True, default='tutorial_cnn')
    parser.add_argument('--out', help='Path of the folder for storing the results', type=str, required=True, default=out_root)
    args = parser.parse_args()

    #train_path = args.train
    #dev_path = args.dev
    #eval_path = args.eval
    preprocessing_alg = args.preprocessing
    preprocessing_axis = args.preprocessing_axis
    out = args.out

    preprocessing_axis = tuple(preprocessing_axis)


    # Set up folders
    if not os.path.isdir(out):
        os.mkdir(out)

    # Compute metrics on eval
    X_eval, y_eval = load_LPC_features('eval')
    reshaped_X_eval = reshape_LPC_features(X_eval)
    norm_X_eval = preprocess_LPC_features(reshaped_X_eval, preprocessing=preprocessing_alg,
                                      preprocessing_axis=preprocessing_axis)

    eval_results = {}
    y_eval_hat = model.predict(np.expand_dims(norm_X_eval, axis=3))[:, 0]

    eer(y_eval, y_eval_hat)
