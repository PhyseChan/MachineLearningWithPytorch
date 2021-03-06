import nltk
import numpy as np
import torch as tr

data_np = np.load("../dataset/quora_pair/train.npy",allow_pickle=True)
print(data_np[0])

def buildWordIdx(dataset):
    word_set = set()
    for items in dataset:
        for item in items:
            for word in item:
                word_set.add(word)
    word2idx = dict(enumerate(word_set))
    idx2word = dict([(i[1],i[0]) for i in enumerate(word2idx.items())])
    return word2idx, idx2word

class NNLM(tr.nn.Module):
    def __init__(self,input_size, repr_size=50):
        self.onehot2rep = tr.nn.Linear(input_size, repr_size)
        self.rep2onehot = tr.nn.Linear(repr_size, input_size)

    def forward(self, x):
        respre