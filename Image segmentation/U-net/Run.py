import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from UnetDataLoader import Unetdataload, PicDataset
from model import SimpleUnet
import torch.nn as nn
import torch.optim as optim
import torch as tr
from torchvision.transforms import Compose, ToTensor, Normalize

train_image_dir = "data/membrane/train/image/"
train_label_dir = "data/membrane/train/label/"
#transformer = Compose([ToTensor, Normalize([0.5, ], [0.5, ])])
unetdata = Unetdataload(train_image_dir, train_label_dir)
plt = unetdata.preview()


Unet_dataset = unetdata.getdataset()
Unet_dataloader = DataLoader(Unet_dataset, batch_size=4, shuffle=True,)
Unet = SimpleUnet()
lr = 0.003
optimizer = optim.Adam(Unet.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()
epochs = 5
for e in range(epochs):
    sum_loss = 0.0
    for _, item in enumerate(Unet_dataloader):
        data, label = item
        output = Unet(data)
        loss = criterion(output.to(dtype=tr.float), label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        sum_loss += loss
        if _ % 4 == 3:
            print(sum_loss / 4)
            sum_loss = 0.0
print()