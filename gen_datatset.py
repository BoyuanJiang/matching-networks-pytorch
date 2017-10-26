# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Created by: BoyuanJiang
# College of Information Science & Electronic Engineering,ZheJiang University
# Email: ginger188@gmail.com
# Copyright (c) 2017

# @Time    :17-8-27 10:18
# @FILE    :gen_datatset.py
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import numpy as np
from scipy import misc
import os

dataset = []
examples = []
# images_background
data_root = "./data/"
alphabets = os.listdir(data_root + "images_background")
for alphabet in alphabets:
    characters = os.listdir(os.path.join(data_root, "images_background", alphabet))
    for character in characters:
        files = os.listdir(os.path.join(data_root, "images_background", alphabet, character))
        examples = []
        for img_file in files:
            img = misc.imresize(
                misc.imread(os.path.join(data_root, "images_background", alphabet, character, img_file)), [28, 28])
            # img = (np.float32(img) / 255.).flatten()
            examples.append(img)
        dataset.append(examples)

# images_evaluation
data_root = "./data/"
alphabets = os.listdir(data_root + "images_evaluation")
for alphabet in alphabets:
    characters = os.listdir(os.path.join(data_root, "images_evaluation", alphabet))
    for character in characters:
        files = os.listdir(os.path.join(data_root, "images_evaluation", alphabet, character))
        examples = []
        for img_file in files:
            img = misc.imresize(
                misc.imread(os.path.join(data_root, "images_evaluation", alphabet, character, img_file)), [28, 28])
            # img = (np.float32(img) / 255.).flatten()
            examples.append(img)
        dataset.append(examples)

np.save(data_root + "dataset.npy", np.asarray(dataset))
