#!/usr/bin/env python3
# coding: utf-8

# [1] https://arxiv.org/abs/2002.02445;
# Using the functions you defined in "utils" and build_model.py, create code to train the model.

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, LearningRateScheduler
from sklearn.preprocessing import OneHotEncoder
from utils.data_loader import load_beam_data
from build_model import build_gru_model

# load training, validation and test data (note: there's no need to scale the data)
train_path = "./dev_dataset_csv/train_set.csv"
val_path = "./dev_dataset_csv/val_set.csv"
test_path = "./viwi_bt_testset_csv_format_eval/testset_evaluation.csv"
Xtr, ytr = load_beam_data(train_path)
Xval, yval = load_beam_data(val_path)
# Xts, yts = load_beam_data(test_path) # Test data is formatted in a differemt way, need to modify the loader
print(f"Training data shape: {Xtr.shape}")
print(f"Validation data shape: {Xval.shape}")
# print(f"Test data shape: {Xts.shape}")
# One-hot-encoding of training and val target
enc = OneHotEncoder()
enc.fit_transform(np.vstack((ytr, [0]))) # needed to manually add codeword "0" in order to one-hot-code to the correct codebook size
# It seems codeword corresponding to index 0 has not been collected in the data
ytr_e = enc.transform(ytr).toarray()
yval_e = enc.transform(yval).toarray()
# yts_e = enc.transform(yts).toarray()
print(f"Encoded training target shape: {ytr_e.shape}")
print(f"Encoded validation target shape: {yval_e.shape}")
# print(f"Encoded test target shape: {yts_e.shape}")

# create the model, print a summary to check all the parameters
K.clear_session()
input_size = Xtr.shape[1]
codebook_size = np.max(Xtr)+1
print(codebook_size)
model = build_gru_model(input_size, int(codebook_size))
print(model.summary())

n_epochs = 100
batch_size = 1000
# compile the model with proper optimizer (Adam(lr=0.001)) and loss function 
init_lr = 1e-3
opt = Adam(lr=init_lr, amsgrad=True)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

# define the following callbacks:
# - model_checkpoint: https://keras.io/api/callbacks/model_checkpoint/ (read doc and understand its function)
# - (optional) reduce_lr: https://keras.io/api/callbacks/reduce_lr_on_plateau/ (read doc and understand its function)
# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001, verbose=1)
lr_update_step = n_epochs // 2
lr_decay = 0.1
def scheduler(epoch, lr):
    if (epoch+1) % lr_update_step == 0:
        print(f"Upating learning rate at epoch: {epoch}; new lr: {lr*lr_decay}")
        return lr*lr_decay
    else:
        return lr        
lr_callback = LearningRateScheduler(scheduler)

# fit model on train data using batch_size and epochs as in paper [1]. Use also the callbacks you defined.
# https://keras.io/api/models/model_training_apis/
hist = model.fit(Xtr, ytr_e, validation_data=(Xval, yval_e), 
                 batch_size=batch_size, epochs=n_epochs, callbacks=[lr_callback])

# plot training statistics. 

# evaluate model on test data. print the accuracy 
