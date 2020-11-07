#!/usr/bin/env python3
# coding: utf-8

# [1] https://arxiv.org/abs/2002.02445;
# using the functions you defined in "utils" and build_model.py, create code to train the model.
# necessary imports
import

# load training, validation and test data (note: there's no need to scale the data)

# build the model, print a summary to check all the parameters

# compile the model with proper optimizer (Adam(lr=0.001)) and loss function 
model.compile()

# define the following callbacks:
# - model_checkpoint: https://keras.io/api/callbacks/model_checkpoint/ (read doc and understand its function)
# - (optional) reduce_lr: https://keras.io/api/callbacks/reduce_lr_on_plateau/ (read doc and understand its function)
# - any other custom or available callback you think might be useful.

# fit model on train data using batch_size and epochs as in paper [1]. Use also the callbacks you defined.
# https://keras.io/api/models/model_training_apis/
train_hist = model.fit()

# plot training statistics. 

# evaluate model on test data. print the accuracy 
