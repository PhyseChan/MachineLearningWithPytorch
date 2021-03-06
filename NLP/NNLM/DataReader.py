import numpy as np
from torch.utils.data import Dataset, DataLoader
import nltk


data_np = np.load("../dataset/quora_pair/train.npy", allow_pickle=True)
print(data_np[0])

def buildWordIdx(dataset):
    word_set = set()
    for items in dataset:
        for item in items:
            for word in item:
                word_set.add(word)
    word2idx = dict(enumerate(word_set))
    idx2word = dict([(i[1], i[0]) for i in enumerate(word2idx.items())])
    return word2idx, idx2word

def buildBow(dataset, num_words):
    dataset = dataset.reshape(1,-1)
    processed_dataset = []
    for sent in dataset:
        if len(sent)<num_words:
            sent = ['<none>'] * (num_words-len(sent)) + sent
        for _,word in enumerate(sent[:-(num_words-1)]):
            processed_dataset.append(sent[_,_+num_words])
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
        return bow_indexing[:-1], bow_indexing[-1]