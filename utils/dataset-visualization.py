#!/usr/bin/env python3
# coding: utf-8

import pandas as pd
from pathlib import Path, PureWindowsPath
import matplotlib.pyplot as plt

train_set_folder = "../dev_dataset_csv/"
train_dataset = pd.read_csv(train_set_folder + "train_set.csv")
print(f"The shape of the training set is: {train_dataset.shape}")

# The training dataset has 281100 training samples. Each row is one data sample. The first 13 columns have a sequence
# of consecutive beams while the last 13 columns have a sequence of paths pointing to 13 consecutive images. The
# first 8 represent the observed beams for a user and the sequence of image where the user appears, and the last 5
# pairs are the label pairs, i.e., they have the future beams of the same user and the corresponding images

# Printing content of the first row (i.e. first data sample of the training dataset)
print(train_dataset.iloc[0, :])

# loading and visualizing the first image of the sequence for the first data sample
image_path = train_dataset.iloc[0, :]["Img Path 1"]
# need to declare the image path as Windows before converting it to a Unix path
path = Path(train_set_folder, PureWindowsPath(image_path))
print(f"The complete path to the image is: {path}")
image = plt.imread(path)  # load the image
print(f"The shape of the image is: {image.shape}")
plt.imshow(image)  # visualize the image
plt.show()


