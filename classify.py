from pandas import read_csv
from conceptnet5.vectors.query import VectorSpaceWrapper
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
import torch
import torch.autograd as auto
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# some globals
STOP_WORDS = set(stopwords.words('english'))
PATH_TO_MINI = '../mini.h5'
TRAIN_SIZE = 0.7
EPOCHS = 10000

# init wrapper and load data
df = read_csv('volunteermatch_data.csv').dropna()
print(df.head())
labels = list(df.columns)[5:]

df[labels] = df[labels] * 1
wrapper = VectorSpaceWrapper(PATH_TO_MINI)
vectorizer = TfidfVectorizer()
vectorizer.fit(df['desc'])

# create + format train and test data
train, test = train_test_split(df, train_size=TRAIN_SIZE, test_size=None)

mean = lambda l: sum(l) / len(l)

def desc2vec(desc):
    tokenized = word_tokenize(desc)
    tokenized = [w for w in tokenized if w not in STOP_WORDS]
    vecs = list(map(lambda s: wrapper.text_to_vector('en', s), tokenized))
    return mean(vecs)

train_xs = np.array(train['desc'].map(desc2vec))
train_xs_with_tfidf = vectorizer.transform(train['desc']).todense()
vocab_size = train_xs_with_tfidf.shape[1]
train_ys = np.array(train[labels])
test_xs = np.array(test['desc'].map(desc2vec))
test_xs_with_tfidf = vectorizer.transform(test['desc']).todense()
test_ys = np.array(test[labels])

# batch training data
batched_xs = []
batched_ys = []
for i in range(0, len(train_xs), 50):
    x = train_xs_with_tfidf[i:i + 50]
    batched_xs.append(np.stack(x))
    y = train_ys[i:i + 50]
    batched_ys.append(np.stack(y))
amt_batches = len(batched_xs)

# define neural network structure and forward pass
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.w1 = nn.Linear(vocab_size, 10)
        self.w2 = nn.Linear(10, 2)

    def forward(self, x):
        x = F.relu(self.w1(x))
        return F.log_softmax(self.w2(x))

# define one model for each label -- perhaps this might work nicely
classifiers = [Net() for label in labels]

# optimizer and loss
adams = [optim.Adam(clf.parameters(), lr=0.01) for clf in classifiers]
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
            preds = clf(to_torch(xb).float())
            loss = cost(preds, to_torch(yb))
            loss.backward()
            adam.step()
            total_loss += loss.item()
        if epoch == EPOCHS - 1:
            final_loss = total_loss / amt_batches
    print(f'Loss for {labels[i]}: {final_loss}')

def make_prediction(x, clfs):
    return np.array([int(clf(to_torch(x).float()).argmax()) for clf in clfs])

def save_model(clfs):
    for i, clf in enumerate(clfs):
        torch.save(clf.state_dict(), f'models/MODEL_{i}.pt')

save_model(classifiers)

# validation step
test_preds = np.stack(list(map(lambda x: make_prediction(x, classifiers),
                               test_xs_with_tfidf)))
acc = accuracy_score(test_preds, test_ys)
print(f'Test accuracy: {acc * 100}%')

