from DataReader import NNLMDataset, buildBow, buildWordIdx
import numpy as np
from torch.utils.data import DataLoader
from NNLMModel import NNLM
import torch.nn as nn
import torch.optim as optim
data_np = np.load("../dataset/Movies/train.npy", allow_pickle=True)
processed_data = buildBow(data_np, num_words=5)
word2idx, idx2word = buildWordIdx(data_np)

nnlmdataset = NNLMDataset(processed_data, word2idx,idx2word)
nnlmdataloader = DataLoader(nnlmdataset, batch_size=200)
word_size = len(word2idx)
model = NNLM(word_size,hidden_size=300)
epochs = 20
lr = 0.1
print_each = 200
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
model.cuda()
for e in range(epochs):
    sum_loss = 0.0
    for _,item in enumerate(nnlmdataloader):
        data,label = item
        output = model(data.cuda())
        loss = criterion(output, label.cuda())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        sum_loss+=loss
        if _%print_each==(print_each-1):
            print(sum_loss/print_each)
            sum_loss=0.0

