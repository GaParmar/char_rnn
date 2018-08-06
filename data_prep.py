# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os
import unicodedata
import string
import pickle
import torch 
import torch.nn as nn
from torch.autograd import Variable


import unidecode
import string
import random
import re

def load_file(path='dataset/input.txt'):
    return unidecode.unidecode(open(path).read());

def get_random_batch(file, batch_size=200):
    start_index = random.randint(0, len(file) - batch_size)
    end_index = start_index + batch_size + 1
    return file[start_index:end_index]

def one_hot_string(string, all_characters):
    tensor = torch.zeros(len(string)).long()
    for c in range(len(string)):
        tensor[c] = all_characters.index(string[c])
    return Variable(tensor)

def random_training_set(file, all_characters):    
    chunk = get_random_batch(file)
    inp = one_hot_string(chunk[:-1], all_characters)
    target = one_hot_string(chunk[1:], all_characters)
    return inp, target