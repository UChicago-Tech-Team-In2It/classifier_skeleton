from conceptnet5.vectors.query import VectorSpaceWrapper
from os import listdir
from sys import argv
from functools import reduce
import pandas as pd
import tensorflow as tf

CSV_PATH = 'scraped_csvs'

def fname_to_tuple(fname):
    '''
        Generates a (DataFrame, label) tuple
    '''
    label = fname[14:-4]
    fname = CSV_PATH + '/' + fname
    df = pd.read_csv(fname)
    df[label] = [True] * len(df)
    return df, label

pairs = list(map(fname_to_tuple, listdir(CSV_PATH)))
labels = list(map(lambda t: t[1], pairs))
dfs = list(map(lambda t: t[0], pairs))

big_df = reduce(lambda df1, df2: df1.merge(df2, left_on=['org', 'title', 'desc'],
                                                right_on=['org', 'title', 'desc'],
                                                how='outer'), dfs)
big_df = big_df.fillna(False)
big_df.to_csv(argv[1])

