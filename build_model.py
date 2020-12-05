#!/usr/bin/env python3
# coding: utf-8

import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Dense, Flatten, Dropout, GRU, LSTM, ConvLSTM2D, MaxPooling3D, MaxPooling2D, BatchNormalization, Concatenate
from tensorflow.keras.activations import relu
from tensorflow.keras.models import Model

'''
Define functions for creating a specific type of network architecture.
In this project, we will implement a first architecture based on GRUs (baseline approach) using only the beam indexes as features.
A second architecture (based on 3D CNN) will additionally exploit camera images to improve the prediction accuracy.
'''
# [1] https://arxiv.org/abs/2002.02445; Github: https://github.com/malrabeiah/VABT/tree/master

# Create a method that returns the first model using keras APIs
def build_gru_model(input_size, codebook_size, embed_size=50, hidden_size=20, num_layers=2, dropout=0.2):

    inputs = Input(shape=(input_size,), name="input_layer")
    # Add an embedding layer (as discussed in paper [1]): https://keras.io/api/layers/core_layers/embedding/
    # The output of the embedding layer should have shape: [batch_size, input_size, embed_size]
    embedding = Embedding(codebook_size, embed_size, name="embedding_layer")(inputs)

    # Add "num_layers" GRU layers with "hidden_size" units. Use the parameters provided in [1]
    # https://keras.io/api/layers/recurrent_layers/gru/
    layer_output = embedding
    for i in range(num_layers):
        if i+1 == num_layers:
            layer_output = GRU(hidden_size, return_sequences=False, 
                               name="recurrent_layer_"+str(i+1))(layer_output)
        else:
            layer_output = GRU(hidden_size, return_sequences=True, 
                               name="recurrent_layer_"+str(i+1))(layer_output)
        # Note: the last layer should have "return_sequences": False, in order to take the output of the last layer. See doc.
        # The final output should have shape [batch_size, hidden_size].
    
    # In the paper, they create a custom layer to extract the output given the # of beams to predict (N).
    # Since for the moment we will focus on the prediction of the first future beam, we shouldn't need to create a similar layer.

    # Add a Dense layer with dimension output_size (i.e. codebook size) and "linear" or "relu" activation function (not clear in their code). Probably relu. 
    # layer_output = relu(layer_output)

    # Add Softmax activation layer.
    out = Dense(codebook_size, activation='softmax')(layer_output)

    model = Model(inputs=inputs, outputs=out)
    return model


# Create a method that returns the second model (CNN+LSTM) using Keras APIs
def build_cnn_model(hist_size, image_shape, codebook_size, num_kernels=40, embed_size=50, hidden_size=20, cnn_layers=3, rnn_layers=2, dropout=0.2):

    '''input_cnn = Input(shape=(hist_size, image_shape[0], image_shape[1], image_shape[2]), name="input_cnn")
    layer_o_cnn = Conv3D(64, kernel_size=(1,3,3), activation="relu", data_format="channels_last", name="conv_1")(input_cnn)
    layer_o_cnn = MaxPooling3D(pool_size=(1,2,2))(layer_o_cnn)
    layer_o_cnn = BatchNormalization(center=True, scale=True)(layer_o_cnn)
    layer_o_cnn = Conv3D(64, kernel_size=(1,3,3), activation="relu", name="conv_2")(layer_o_cnn)
    layer_o_cnn = MaxPooling3D(pool_size=(1,2,2))(layer_o_cnn)
    layer_o_cnn = BatchNormalization(center=True, scale=True)(layer_o_cnn)
    layer_o_cnn = Conv3D(64, kernel_size=(1,3,3), activation="relu", name="conv_3")(layer_o_cnn)
    layer_o_cnn = MaxPooling3D(pool_size=(1,2,2))(layer_o_cnn)
    layer_o_cnn = BatchNormalization(center=True, scale=True)(layer_o_cnn)'''

    input_cnn = Input(shape=(hist_size, image_shape[0], image_shape[1], image_shape[2]), name="input_cnn_lstm")
    input_rnn = Input(shape=(hist_size,), name="input_rnn")

    # CNN+LSTM part of the network (spatio-temporal features extraction from a sequence of images)
    layer_out_cnn = input_cnn
    for i in range(cnn_layers):
        if i+1 == cnn_layers:
            layer_out_cnn = ConvLSTM2D(hidden_size, strides=(2,2), kernel_size=(3,3), return_sequences=False, data_format="channels_last",
                                       name="cnn_layer_"+str(i+1))(layer_out_cnn)
            layer_out_cnn = MaxPooling2D(pool_size=(2,2))(layer_out_cnn)
            layer_out_cnn = BatchNormalization()(layer_out_cnn)
        else:
            layer_out_cnn = ConvLSTM2D(num_kernels, strides=(2,2), kernel_size=(3,3), return_sequences=True, data_format="channels_last",
                                       name="cnn_layer_"+str(i+1))(layer_out_cnn)
            layer_out_cnn = MaxPooling3D(pool_size=(1,2,2))(layer_out_cnn)
            layer_out_cnn = BatchNormalization()(layer_out_cnn)
    layer_out_cnn = Flatten()(layer_out_cnn)

    # LSTM part of the network (temporal features extraction from a sequence of beams)
    embedding = Embedding(codebook_size, embed_size, name="embedding_layer")(input_rnn)
    layer_out_rnn = embedding
    for i in range(rnn_layers):
        if i+1 == rnn_layers:
            layer_out_rnn = LSTM(hidden_size, return_sequences=False, 
                                 name="recurrent_layer_"+str(i+1))(layer_out_rnn)
        else:
            layer_out_rnn = LSTM(hidden_size, return_sequences=True, 
                                 name="recurrent_layer_"+str(i+1))(layer_out_rnn)
    # Merge the two networks
    merge_out = Concatenate()([layer_out_rnn, layer_out_cnn])

    dense_out = Dense(codebook_size, activation='relu')(merge_out)
    dense_out = Dense(codebook_size, activation='softmax')(dense_out)

    model = Model(inputs=[input_rnn, input_cnn], outputs=dense_out)

    return model