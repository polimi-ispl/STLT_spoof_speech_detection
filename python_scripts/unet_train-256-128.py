#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np
import sys
import os
import scipy.stats as stats
import time
import tqdm

import tensorflow as tf
import tensorflow.keras

from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D , Flatten
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


# ### UNET definition

# In[8]:


os.environ["CUDA_VISIBLE_DEVICES"]="3"


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


# In[9]:


gpus


# In[10]:


# definire la UNET

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

# In[11]:


train_txt_path = '../dataset/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt'

df_train = pd.read_csv(train_txt_path, sep=" ", header=None)
df_train.columns = ["speaker_id", "audio_filename", "null", "system_id", "label"]
df_train = df_train.drop(columns="null")

dev_txt_path = '../dataset/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt'

df_dev = pd.read_csv(dev_txt_path, sep=" ", header=None)
df_dev.columns = ["speaker_id", "audio_filename", "null", "system_id", "label"]
df_dev = df_dev.drop(columns="null")

eval_txt_path = '../dataset/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt'

df_eval = pd.read_csv(eval_txt_path, sep=" ", header=None)
df_eval.columns = ["speaker_id", "audio_filename", "null", "system_id", "label"]
df_eval = df_eval.drop(columns="null")


# ### Params

# In[12]:


# i parametri da testare sono:
# nfft = 64, hop_size = 32 fatto
# nfft = 128, hop_size = 64 fatto
# nfft = 256, hop_size = 128 fatto
# nffr = 512, hop_size = 256

nfft = 256
hop_size = 128

alg_list = ['A01', 'A02', 'A03', 'A04', 'A05', 'A06']

epochs = 100
batch_size = 64

input_size = nfft // 2


# ### Train UNET

# In[ ]:


train_feat_root_path = '../features/bicoherences/train_nfft_{}_hop_size_{}'.format(nfft, hop_size)


for alg in alg_list:
    alg_df_train = df_train[df_train['system_id']==alg]
    
    unet_input = []
    
    for index, row in tqdm.tqdm(alg_df_train.iterrows(), total=alg_df_train.shape[0]):
        feat_path = os.path.join(train_feat_root_path, row['audio_filename'] + '.npy')
        bicoh = np.load(feat_path)
        mag = np.abs(bicoh)
        phase = np.angle(bicoh)
        
        # qui puoi concatenare le magnitude per l'input alla rete nella variabile unet_input
        unet_input.append(mag)
        
    unet_input = np.array(unet_input)
    unet_input = unet_input[..., np.newaxis]
    
    train_validation_index = int(np.round(unet_input.shape[0] * 0.75))
    X_train = unet_input[:train_validation_index]
    X_valid = unet_input[train_validation_index:]

    model_checkpoint_name = '../features/unet/models/train_nfft_{}_hop_size_{}_alg_{}.ckpt'.format(
        nfft, hop_size, alg)
    
    history_name = '../features/unet/history/train_nfft_{}_hop_size_{}_alg_{}.npy'.format(
        nfft, hop_size, alg)
    
    
    
    model, encoder = unet(None, (input_size, input_size, 1))

    checkpoint = ModelCheckpoint(model_checkpoint_name, 
                                 monitor='val_loss', 
                                 save_best_only=True, 
                                 save_weights_only=True, 
                                 mode='auto',
                                 verbose=0)
    
    early = EarlyStopping(monitor='val_loss', min_delta=0, patience=5,
                          verbose=0, mode='auto')


    history = model.fit(X_train, X_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    shuffle=True,
                    validation_data=(X_valid, X_valid),
                    callbacks=[ checkpoint, early])
    np.save(history_name, history.history)
    

    


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # Feature computation

# ### Load models

# In[ ]:





# ### Compute features

# train_feat_root_path = '../features/bicoherences/train_nfft_{}_hop_size_{}'.format(nfft, hop_size)
# 
# for alg in alg_list:
#     
#     # carichiamo il modello corrispondente dalla cartella 
#     model_folder = '../features/unet/models/train_nfft_{}_hop_size_{}_alg_{}'.format(nfft, hop_size, alg)
# 
#     # creiamo un campo in un dataframe per ora vuoto
#     feat_name = 'unet_mse_nfft_{}_hop_size_{}_alg_{}'.format(nfft, hop_size, alg)
#     df_train[feat_name] = np.nan
#     
#     break
#     for index, row in tqdm.tqdm(df_train.iterrows(), total=df_train.shape[0]):
#         feat_path = os.path.join(train_feat_root_path, row['audio_filename'] + '.npy')
#         bicoh = np.load(feat_path)
#         mag = np.abs(bicoh)
#         phase = np.angle(bicoh)
#         
#         # calcoliamo l'MSE 
#         
#         # salviamo l'MSE nel corrispondente campo
#         
#         # df_train.at[index, feat_name] = mse
#         
# 

# dev_feat_root_path = '../features/bicoherences/dev_nfft_{}_hop_size_{}'.format(nfft, hop_size)
# 
# for alg in alg_list:
#     
#     # carichiamo il modello corrispondente dalla cartella 
#     model_folder = '../features/unet/models/train_nfft_{}_hop_size_{}_alg_{}'.format(nfft, hop_size, alg)
# 
#     # creiamo un campo in un dataframe per ora vuoto
#     feat_name = 'unet_mse_nfft_{}_hop_size_{}_alg_{}'.format(nfft, hop_size, alg)
#     df_train[feat_name] = np.nan
#     
#     break
#     for index, row in tqdm.tqdm(df_train.iterrows(), total=df_train.shape[0]):
#         feat_path = os.path.join(train_feat_root_path, row['audio_filename'] + '.npy')
#         bicoh = np.load(feat_path)
#         mag = np.abs(bicoh)
#         phase = np.angle(bicoh)
#         
#         # calcoliamo l'MSE 
#         
#         # salviamo l'MSE nel corrispondente campo
#         
#         # df_train.at[index, feat_name] = mse

# train_feat_root_path = '../features/bicoherences/train_nfft_{}_hop_size_{}'.format(nfft, hop_size)
# 
# for alg in alg_list:
#     
#     # carichiamo il modello corrispondente dalla cartella 
#     model_folder = '../features/unet/models/train_nfft_{}_hop_size_{}_alg_{}'.format(nfft, hop_size, alg)
# 
#     # creiamo un campo in un dataframe per ora vuoto
#     feat_name = 'unet_mse_nfft_{}_hop_size_{}_alg_{}'.format(nfft, hop_size, alg)
#     df_train[feat_name] = np.nan
#     
#     break
#     for index, row in tqdm.tqdm(df_train.iterrows(), total=df_train.shape[0]):
#         feat_path = os.path.join(train_feat_root_path, row['audio_filename'] + '.npy')
#         bicoh = np.load(feat_path)
#         mag = np.abs(bicoh)
#         phase = np.angle(bicoh)
#         
#         # calcoliamo l'MSE 
#         
#         # salviamo l'MSE nel corrispondente campo
#         
#         # df_train.at[index, feat_name] = mse

# In[ ]:




