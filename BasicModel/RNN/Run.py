from Util import GenerateBinarydata
from torch.utils.data import DataLoader
from BinaryDataLoader import BinaryDataSet
import torch.optim as optim
import torch.nn as nn
from RNN import RNN
import torch

dataGenerator = GenerateBinarydata()
data = dataGenerator()
dataset = BinaryDataSet(data, cat=32)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
epochs = 200
a0 = torch.randn((2 ** 5))
model = RNN(2, 32, a0)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.004)

for e in range(epochs):
    sum_loss = 0.0
    for _, item in enumerate(dataloader):
        data, label = item
        output = model(data)
        loss = criterion(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        sum_loss += loss
        if _ % 5 == 4:
            print("loss:", sum_loss / 5)
            sum_loss = 0.0

model.eval()
data_test = dataGenerator()
dataset_test = BinaryDataSet(data_test, cat=32)
dataloader_test = DataLoader(dataset_test, batch_size=40, shuffle=True)
data_test, label_test = next(iter(dataloader_test))
res = model(data_test)
print((torch.argmax(res,dim=1)==label_test).nonzero().shape[0]/40)
print(torch.argmax(res,dim=1),label_test)
