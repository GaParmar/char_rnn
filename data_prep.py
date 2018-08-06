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
    # special case if <EOS> is encountered
    if string == "<EOS>":
        tensor = torch.zeros(1,1,len(all_characters)+1)
        tensor[0][0][len(all_characters)] = 1
        return tensor 
    
    tensor = torch.zeros(len(string), 1, len(all_characters)+1)
    for c in range(len(string)):
        char = string[c]
        tensor[c][0][all_characters.index(char)] = 1
    return tensor

def indexify_string(string, all_characters):
    indices = []
    for i in range(len(string)):
        ch = string[i]
        indices.append(all_characters.index(ch))
    indices.append(len(all_characters)) #EOS token
    return torch.LongTensor(indices)

def random_training_set(file, all_characters):    
    chunk = get_random_batch(file)
    inp = one_hot_string(chunk[:-1], all_characters)
    target = indexify_string(chunk[1:], all_characters)
    return inp, target