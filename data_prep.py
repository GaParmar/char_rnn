# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os
import unicodedata
import string
import pickle

all_letters = string.ascii_letters + " .,;'-"
n_letters = len(all_letters) + 1 # Plus EOS marker

def findFiles(path): 
    return glob.glob(path)

# Name: unicodeToAscii
# Description: Converts a unicode string to ascii
# Parameters: s - string to be converted
# Return: converted string
# https://stackoverflow.com/questions/517923/what-is-the-best-way-to-remove-accents-in-a-python-unicode-string/518232#518232
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )


# Name: readLines
# Description: open a file and split into lines
# Parameters: filename - the path to the file
# Return: array of lines
def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    a_lines = []
    for l in lines:
        a_lines.append(unicodeToAscii(l))
    return a_lines


def buildCategories(dataset_path):
    all_categories = []
    category_map = {}
    for filename in findFiles(dataset_path+"/names/*.txt"):
        category = os.path.splitext(os.path.basename(filename))[0]
        all_categories.append(category)
        category_map[category] = readLines(filename)
    return all_categories, category_map


def save_dataset(map_cat, filename=".categories_map.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(map_cat, f)

def load_dataset(filename=".categories_map.pkl"):
    with open(filename, "rb") as f:
        categories_map = pickle.load(f)
    return categories_map
