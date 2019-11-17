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

PATH_TO_MINI = '../mini.h5'
TRAIN_SIZE = 0.7

df = read_csv('all.csv')
labels = list(df.columns)[4:]
df[labels] = df[labels] * 1
wrapper = VectorSpaceWrapper(PATH_TO_MINI)

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

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.w1 = nn.Linear(300, 50)
        self.w2 = nn.Linear(50, 50)
        self.w3 = nn.Linear(50, 11)

    def forward(self, x):
        x = F.relu(self.w1(x))
        x = F.relu(self.w2(x))
        return F.sigmoid(self.w3(x))
    
net = Net()
cost = nn.BCEWithLogitsLoss()
adam = optim.Adam(net.parameters(), lr=0.0001)

to_torch = lambda x: auto.Variable(torch.from_numpy(x))

amt_samples = len(train_xs)

for epoch in range(1000):
    running_loss = 0.0
    for xs, ys in zip(batched_xs, batched_ys):
        adam.zero_grad()
        
        preds = net(to_torch(xs))
        loss = cost(preds, to_torch(ys).float())
        loss.backward()
        adam.step()

        running_loss += loss.item()
    print(f'Loss: {running_loss / amt_samples}')

print('Training done.')

print('Validating on Test set...')
for x, y in zip(test_xs, test_ys):
    pred = net(to_torch(x))
    print(pred)
    print(y)





