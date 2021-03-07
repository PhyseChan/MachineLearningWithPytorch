import numpy as np
from torch.utils.data import Dataset, DataLoader
import nltk
import torch
# data_np = np.load("../dataset/Movies/train.csv", allow_pickle=True)
# print(data_np[0])


def buildWordIdx(dataset):
    word_set = set()
    for item in dataset:
        for word in item:
            word_set.add(word)
    word_set.add('<none>')
    idx2word = dict(enumerate(word_set))
    word2idx = dict([ (item,_) for _,item in enumerate(word_set)])
    return word2idx, idx2word


def buildBow(dataset, num_words):
    processed_dataset = []
    for sent in dataset:
        if len(sent) < num_words:
            sent = ['<none>'] * (num_words - len(sent)) + sent
        for _, word in enumerate(sent[:-(num_words - 1)]):
            processed_dataset.append(sent[_: _ + num_words])
    return processed_dataset


class NNLMDataset(Dataset):
    def __init__(self, data, word2idx, idx2word):
        self.data = data
        self.word2idx = word2idx

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        bow = self.data[idx]
        bow_indexing = [self.word2idx[word] for word in bow]
        return torch.tensor(bow_indexing[:-1]), torch.tensor(bow_indexing[-1])
