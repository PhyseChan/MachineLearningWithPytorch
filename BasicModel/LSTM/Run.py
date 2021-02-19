from MinstDataLoader import MnistReader
import torch.optim as optim
import torch.nn as nn
from LSTM import LSTM
import torch
import numpy as np

training_loader, test_loader = MnistReader().dataloader(batch_size_trian= 20)

epochs = 1
c0 = torch.randn(1, 20, 10)
h0 = torch.randn(1,20, 10)
model = LSTM(28, 64,10)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
for e in range(epochs):
    sum_loss = 0.0
    for _, item in enumerate(training_loader):
        data, label = item
        data = torch.squeeze(data, dim=1)
        output = model(data, h0, c0)
        loss = criterion(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        sum_loss += loss
        if _ % 100 == 99:
            print(f'loss: {sum_loss.item() / 100:.2f}')
            sum_loss=0.0
