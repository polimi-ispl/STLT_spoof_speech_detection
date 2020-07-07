import argparse
import tensorflow.keras
import os
from LPC_params import *
from LPC_features_utils import *
import cnn_utils
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('--train', help='Path of the dataset used for training', type=str, required=True)
    #parser.add_argument('--dev', help='Path of the dataset used for development', type=str, required=True)
    #parser.add_argument('--eval', help='Path of the dataset used for evaluation', type=str, required=True)

    parser.add_argument('--preprocessing', help='Preprocessing algorithm', type=str, required=True, default='l2_norm')
    parser.add_argument('--preprocessing_axis', help='Axis used for preprocessing operation (among all samples or along lpc axis', type=int, nargs='+', required=True, default='1')
    parser.add_argument('--cnn_model', help='CNN model', type=str, required=True, default='tutorial_cnn')
    args = parser.parse_args()

    #train_path = args.train
    #dev_path = args.dev
    #eval_path = args.eval
    preprocessing_alg = args.preprocessing
    preprocessing_axis = args.preprocessing_axis

    cnn_model = args.cnn_model

    # Set up training name and folder

    model_name = "model-{}_preprocessing-{}_axis-{}".format(cnn_model, preprocessing_alg, str(preprocessing_axis).strip('[]'))
    train_root = os.path.join(project_root, model_name)

    if not os.path.isdir(tensorboard_logs_root):
         os.mkdir(tensorboard_logs_root)

    if not os.path.isdir(train_root):
         os.mkdir(train_root)

    # Set up folder
    # if not os.path.isdir(tensorboard_logs_root):
    #     os.mkdir(tensorboard_logs_root)
    #
    # if not os.path.isdir(checkpoint_root):
    #     os.mkdir(checkpoint_root)
    #
    # if not os.path.isdir(history_root):
    #     os.mkdir(history_root)

    # Fix axis arguments, from list to tuple
    preprocessing_axis = tuple(preprocessing_axis)

    # Set up  callbacks
    tensorboard_log_path = os.path.join(tensorboard_logs_root, model_name)
    tensorboard_callback = tensorflow.keras.callbacks.TensorBoard(log_dir=tensorboard_log_path)

    reduce_lr = tensorflow.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, min_lr=0.001)

    checkpoint_path = os.path.join(train_root, "checkpoint.hdf5")
    model_checkpoint = tensorflow.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=0, save_best_only=True)

    # Load data
    X_train, y_train = load_LPC_features('train')
    X_dev, y_dev = load_LPC_features('dev')

    # Reshape data
    reshaped_X_train = reshape_LPC_features(X_train)
    reshaped_X_dev = reshape_LPC_features(X_dev)

    # Preprocess data
    preproc_params = {preprocessing_alg: [None, None]}

    # If I am not normalizing along the axis 0 (the samples), I can't use the same parameters for training and dev

    norm_X_train, preproc_params = preprocess_LPC_features(reshaped_X_train, p_axis=preprocessing_axis, **preproc_params)

    if 0 not in preprocessing_axis:
        preproc_params = {preprocessing_alg: [None, None]}

    norm_X_dev, _ = preprocess_LPC_features(reshaped_X_dev, p_axis=preprocessing_axis,**preproc_params)

   # np.save(os.path.join(preprocessing_root, 'preprocessing_' +  model_name +'.npy'), preproc_params)

    np.save(os.path.join(train_root, "preprocessing_params.npy"), preproc_params)

    y_train_cat = tensorflow.keras.utils.to_categorical(y_train, num_classes)
    y_dev_cat = tensorflow.keras.utils.to_categorical(y_dev, num_classes)

    # Load the model
    model = getattr(cnn_utils, cnn_model)()

    #print(model.summary())

    # Compile the model

    model.compile(loss=tensorflow.keras.losses.categorical_crossentropy,
              optimizer=tensorflow.keras.optimizers.Adam(),
              metrics=['accuracy'])

    # Train the model
    history = model.fit(np.expand_dims(norm_X_train, axis=3), y_train_cat,
                                     batch_size=batch_size,
                                     epochs=epochs,
                                     verbose=1,
                                     callbacks=[reduce_lr, tensorboard_callback, model_checkpoint],
                                     validation_data=[np.expand_dims(norm_X_dev, axis=3), y_dev_cat])

    # Save history

    history_path = os.path.join(train_root, "history.npz")
    np.savez(history_path, history=history.history, params=history.params, epochs=history.epoch)
