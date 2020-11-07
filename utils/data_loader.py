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

def load_beam_data():
    # you can change name of this function and add other functions, if the code will be much cleaner, of course