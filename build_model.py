#!/usr/bin/env python3
# coding: utf-8

import tensorflow as tf
# complete with all the necessary imports

'''
Define functions for creating a specific type of network architecture.
In this project, we will implement a first architecture based on GRUs (baseline approach) using only the beam indexes as features.
A second architecture (based on 3D CNN) will additionally exploit camera images to improve the prediction accuracy.
'''
# [1] https://arxiv.org/abs/2002.02445; Github: https://github.com/malrabeiah/VABT/tree/master

# Create a method that returns the first model using keras APIs
def build_gru_model(input_size, output_size, embed_size, hidden_size, num_layers):
    model = Sequential()
    # Add an embedding layer (as discussed in paper [1]): https://keras.io/api/layers/core_layers/embedding/
    # The output of the embedding layer should have shape: [batch_size, input_size, embed_size]
    model.add()

    # Add "num_layers" GRU layers with "hidden_size" units. Use the parameters provided in [1]
    # https://keras.io/api/layers/recurrent_layers/gru/
    for i in range(num_layers):
        model.add()
        # Note: the last layer should have "return_sequences": True, in order to take the output of the last layer. See doc.
        # The final output should have shape [batch_size, hidden_size]. Check with model.summary(), I might be wrong.
    
    # In the paper, they create a custom layer to extract the output given the # of beams to predict (N).
    # Since for the moment we will focus on the prediction of the first future beam, we shouldn't need to create a similar layer.

    # Add a Dense layer with dimension output_size (i.e. codebook size) and "linear" or "relu" activation function (not clear in their code). Probably relu. 
    model.add()

    # Add Softmax activation layer. Be careful: this is not another Dense layer with activation function "softmax" (introduces
    # new training parameters), but just a layer which computes softmax activation on the output of the previous layer.
    # See Keras Doc.

    return model


# Create a method that returns the second model (3D CNNs + GRUs) using Keras APIs
