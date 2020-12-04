#!/usr/bin/env python3
# coding: utf-8

import pandas as pd
from pathlib import Path, PureWindowsPath

'''
Define a function for loading the data from a given folder path. 
(note: in this project, training, validation and test datasets are located in different folders)
'''
# You should write some code based on the content of "dataset-visualization.py"

# To make things easier, first, create a function to load data for the first problem (beam prediction using past beams).
# Have clear the dataset composition before doing this.
# The 9th column (starting from 1) would be the true labels column (we want to predict the first future beam)
# You should one-hot-encode the true labels columns. There's an sklearn function for this.

def load_beam_data(path):
    df = pd.read_csv(path)
    feature_cols = ["Beam 1", "Beam 2", "Beam 3", "Beam 4", "Beam 5", "Beam 6", "Beam 7", "Beam 8"]
    target_cols = ["Beam 9"]
    features = df[feature_cols].to_numpy()
    target = df[target_cols].to_numpy()
    return features, target 

def load_beam_visual_data(path):
    df = pd.read_csv(path)
    df.head()
    image_cols = ["Img Path 1", "Img Path 2", "Img Path 3", "Img Path 4", "Img Path 5", "Img Path 6", "Img Path 7", "Img Path 8"]
    beam_cols = ["Beam 1", "Beam 2", "Beam 3", "Beam 4", "Beam 5", "Beam 6", "Beam 7", "Beam 8"]
    target_cols = ["Beam 9"]
    target = df[target_cols].to_numpy()
    
    image_path = train_dataset.iloc[0, :]["Img Path 1"]
    # need to declare the image path as Windows before converting it to a Unix path
    path = Path(train_set_folder, PureWindowsPath(image_path))