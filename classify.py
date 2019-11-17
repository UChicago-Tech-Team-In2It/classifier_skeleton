from pandas import read_csv
from conceptnet5.vectors.query import VectorSpaceWrapper
from sklearn.model_selection import train_test_split
from wordfreq import simple_tokenize
import numpy as np
import torch

PATH_TO_MINI = '../mini.h5'
TRAIN_SIZE = 0.7

df = read_csv('all.csv')
wrapper = VectorSpaceWrapper(PATH_TO_MINI)

train, test = train_test_split(df, train_size=TRAIN_SIZE, test_size=None)

mean = lambda l: sum(l) / len(l)

def desc2vec(desc):
    tokenized = simple_tokenize(desc)
    vecs = list(map(lambda s: wrapper.text_to_vector('en', s), tokenized))
    return mean(vecs)


