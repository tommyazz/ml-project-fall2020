#!/usr/bin/env python3
# coding: utf-8

# [1] https://arxiv.org/abs/2002.02445;
# Using the functions you defined in "utils" and build_model.py, create code to train the model.

import numpy as np
import pickle
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, LearningRateScheduler, ModelCheckpoint
from sklearn.preprocessing import OneHotEncoder
from utils.data_loader import *
from build_model import build_cnn_model

class CustomDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, X, y, hist_size, im_size, batch_size=32, shuffle=True):
        self.X = X
        self.y = y
        self.tot_data = self.X.shape[0]
        self.num_classes = self.y.shape[1]
        self.hist_size = hist_size
        self.im_size = im_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()
    
    def __len__(self):
        # returns the number of steps per epoch
        return self.tot_data // self.batch_size

    def __getitem__(self, index):
        # returns one batch of data
        indexes = self.index[index * self.batch_size:(index + 1) * self.batch_size]
        return self.__getdata__(indexes)

    def __getdata__(self, indexes):
        beam_batch = []
        image_batch = []
        y_batch = []
        scaling = tf.constant([255], dtype=tf.dtypes.float32)
        for i,idx in enumerate(indexes):
            y_batch.append(self.y[idx,:])
            beam_batch.append(self.X[idx, 0:self.hist_size])
            h_image_b = []
            for j in range(self.hist_size):
                img = tf.image.decode_jpeg(tf.io.read_file(tf.compat.path_to_str(self.X[idx, self.hist_size+j])))
                img = tf.image.resize(img, [self.im_size[0], self.im_size[1]])
                h_image_b.append(tf.divide(img, scaling))
            image_batch.append(h_image_b)
        return [tf.stack(beam_batch), tf.stack(image_batch)], tf.stack(y_batch)

    def on_epoch_end(self):
        self.index = np.arange(self.tot_data)
        if self.shuffle:
            np.random.shuffle(self.index)

# load training, validation and test data (note: there's no need to scale the data)

train_path = "./dev_dataset_csv/train_set.csv"
val_path = "./dev_dataset_csv/val_set.csv"
test_path = "./viwi_bt_testset_csv_format_eval/testset_evaluation.csv"
Xtr, ytr = load_beam_visual_data(train_path)
Xval, yval = load_beam_visual_data(val_path)
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
hist_size = Xtr.shape[1]//2
codebook_size = ytr_e.shape[1]
print(codebook_size)
target_im_size = (1280//4,720//4,3)

model = build_cnn_model(hist_size, target_im_size, int(codebook_size), num_kernels=20)
print(model.summary())

# compile the model with proper optimizer (Adam(lr=0.001)) and loss function 
init_lr = 1e-3
opt = Adam(lr=init_lr, amsgrad=True)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

# define the following callbacks:
# - model_checkpoint: https://keras.io/api/callbacks/model_checkpoint/ (read doc and understand its function)
'''
def scheduler(epoch, lr):
    if (epoch+1) % lr_update_step == 0:
        print(f"Upating learning rate at epoch: {epoch}; new lr: {lr*lr_decay}")
        return lr*lr_decay
    else:
        return lr        
lr_callback = LearningRateScheduler(scheduler)
'''

model_path = "./model-{epoch:02d}.h5"
model_checkpoint = ModelCheckpoint(model_path, monitor="val_accuracy", save_best_only=True, verbose=1)

n_epochs = 100
tr_batch_size = 32
val_batch_size = 100
# Creates Training and Validation data generators
train_generator = CustomDataGenerator(Xtr, ytr_e, hist_size, target_im_size, batch_size=tr_batch_size)
val_generator = CustomDataGenerator(Xval, yval_e, hist_size, target_im_size, batch_size=val_batch_size)
[Xval_beam, Xval_image], yval_gen = val_generator.__getitem__(0)

# fit model on train data using batch_size and epochs as in paper [1]. Use also the callbacks you defined.
# https://keras.io/api/models/model_training_apis/
hist = model.fit(train_generator, validation_data=([Xval_beam, Xval_image], yval_gen), epochs=n_epochs, callbacks=[model_checkpoint])
# hist = model.fit(train_generator, epochs=n_epochs)

# plot training statistics. 
pickle.dump(hist.history, open( "history.p", "wb" ))

# evaluate model on test data. print the accuracy 