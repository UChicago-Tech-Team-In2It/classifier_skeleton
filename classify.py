from pandas import read_csv
from conceptnet5.vectors.query import VectorSpaceWrapper
from sklearn.model_selection import train_test_split
from wordfreq import simple_tokenize
import numpy as np
import torch
import torch.autograd as auto
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# some globals
PATH_TO_MINI = '../mini.h5'
TRAIN_SIZE = 0.7
EPOCHS = 1000

# init wrapper and load data
df = read_csv('all.csv')
labels = list(df.columns)[4:]
df[labels] = df[labels] * 1
wrapper = VectorSpaceWrapper(PATH_TO_MINI)

# create + format train and test data
train, test = train_test_split(df, train_size=TRAIN_SIZE, test_size=None)

mean = lambda l: sum(l) / len(l)

def desc2vec(desc):
    tokenized = simple_tokenize(desc)
    vecs = list(map(lambda s: wrapper.text_to_vector('en', s), tokenized))
    return mean(vecs)

train_xs = train['desc'].map(desc2vec)
train_ys = np.array(train[labels])
test_xs = test['desc'].map(desc2vec)
test_ys = np.array(test[labels])

# batch training data
batched_xs = []
batched_ys= []
for i in range(0, len(train_xs), 50):
    x = train_xs[i:i + 50]
    batched_xs.append(np.stack(x))
    y = train_ys[i:i + 50]
    batched_ys.append(np.stack(y))
amt_batches = len(batched_xs)

# define neural network structure and forward pass
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.w1 = nn.Linear(300, 50)
        self.w2 = nn.Linear(50, 50)
        self.w3 = nn.Linear(50, 11)

    def forward(self, x):
        x = F.relu(self.w1(x))
        x = F.relu(self.w2(x))
        return F.log_softmax(self.w3(x))

# define one model for each label -- perhaps this might work nicely
classifiers = [Net() for label in labels]

# optimizer and loss
adams = [optim.Adam(clf.parameters(), lr=0.001) for clf in classifiers]
cost = nn.CrossEntropyLoss()

# train each classifier now
to_torch = lambda x: auto.Variable(torch.from_numpy(x))

for i, clf in enumerate(classifiers):
    print(f'Training model on {labels[i]}...')
    adam = adams[i]
    final_loss = 0
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for xb, yb in zip(batched_xs, batched_ys):
            yb = np.array([l[i] for l in yb])
            adam.zero_grad()
            preds = clf(to_torch(xb))
            loss = cost(preds, to_torch(yb))
            loss.backward()
            adam.step()
            total_loss += loss.item()
        if epoch == EPOCHS - 1:
            final_loss = total_loss / amt_batches
    print(f'Loss for {labels[i]}: {final_loss}')

# validation step




