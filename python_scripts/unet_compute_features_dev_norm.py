#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import sys
import os
import scipy.stats as stats
import time
from tqdm import tqdm

import tensorflow as tf
import tensorflow.keras
import gc

from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D , Flatten
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


# ### UNET definition

# In[2]:



os.environ["CUDA_VISIBLE_DEVICES"]="4"

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


# In[3]:


gpus


# In[4]:


def unet(pretrained_weights=None, input_size=(256, 256, 1)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=conv10)

    model.compile(optimizer=Adam(lr=1e-4), loss='mean_squared_error', metrics=['accuracy'])
    encoder=Model(inputs=inputs, outputs=drop5)
    # model.summary()

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model, encoder


# ### Load CSV

# In[5]:





# ### Params

# In[6]:


# i parametri da testare sono:
# nfft = 64, hop_size = 32 
# nfft = 128, hop_size = 64 fatto
# nfft = 256, hop_size = 128 
# nfft = 512, hop_size = 256

nfft = [512, 256, 128, 64]
hop_size = [256, 128, 64, 32]
fft_params = zip(nfft, hop_size)

alg_list = ['A01', 'A02', 'A03', 'A04', 'A05', 'A06']

epochs = 100
batch_size = 4 #=16 per nfft > 512

dev_txt_path = '../dataset/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt'

for f in fft_params:
    print('FFT params {}'.format(f))

    nfft = f[0]
    hop_size = f[1]
    input_size = nfft // 2

    df_dev = pd.read_csv(dev_txt_path, sep=" ", header=None)
    df_dev.columns = ["speaker_id", "audio_filename", "null", "system_id", "label"]
    df_dev = df_dev.drop(columns="null")

    dev_feat_root_path = '../features/bicoherences/dev_nfft_{}_hop_size_{}'.format(nfft, hop_size)

    for alg in alg_list:

        model_folder = '../features/unet_norm/models/train_nfft_{}_hop_size_{}_alg_{}.ckpt'.format(
            nfft, hop_size, alg)

        model, encoder = unet(model_folder, (input_size, input_size, 1))

        # creiamo un campo in un dataframe per ora vuoto
        feat_name = 'unet_mse_alg_{}'.format(alg)
        df_dev[feat_name] = np.nan

        split_dataframe_index = len(df_dev) // 2
        df_dev_1 = df_dev.iloc[:split_dataframe_index, :]
        df_dev_2 = df_dev.iloc[split_dataframe_index: , :]

        mag_volume = []
        print("First half of dev")
        for index, row in tqdm(df_dev_1.iterrows(), total=len(df_dev_1)):
            feat_path = os.path.join(dev_feat_root_path, row['audio_filename'] + '.npy')
            bicoh = np.load(feat_path)
            mag = np.abs(bicoh)
            mag = (mag - np.min(mag)) / (np.max(mag) - np.min(mag))

            mag_volume.append(mag)

        mag_volume = np.array(mag_volume)
        mag_volume = mag_volume[..., np.newaxis]

        output = model.predict(mag_volume, batch_size=batch_size)
        compressed = encoder.predict(mag_volume, batch_size=batch_size)
        output_compressed = encoder.predict(output, batch_size=batch_size)

        mse = np.mean(np.square(compressed - output_compressed), axis=(1, 2, 3))
        df_dev.at[:split_dataframe_index - 1 , feat_name] = mse

        print("Second half of dev")
        mag_volume = []
        for index, row in tqdm(df_dev_2.iterrows(), total=len(df_dev_2)):
            feat_path = os.path.join(dev_feat_root_path, row['audio_filename'] + '.npy')
            bicoh = np.load(feat_path)
            mag = np.abs(bicoh)
            mag = (mag - np.min(mag)) / (np.max(mag) - np.min(mag))

            mag_volume.append(mag)

        mag_volume = np.array(mag_volume)
        mag_volume = mag_volume[..., np.newaxis]

        output = model.predict(mag_volume, batch_size=batch_size)
        compressed = encoder.predict(mag_volume, batch_size=batch_size)
        output_compressed = encoder.predict(output, batch_size=batch_size)

        mse = np.mean(np.square(compressed - output_compressed), axis=(1, 2, 3))

        df_dev.at[split_dataframe_index:, feat_name] = mse

        del model
        del encoder
        gc.collect()
        tf.keras.backend.clear_session()
    df_dev.to_pickle('../features/unet_norm/dev_nfft_{}_hop_size_{}.pkl'.format(nfft, hop_size))





