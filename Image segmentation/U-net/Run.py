from UnetDataLoader import Unetdataload, PicDataset
from unet_model.UNET import Unet
import torch.optim as optim
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import matplotlib.image as image
import os
import numpy as np
train_image_dir = "data/membrane/train/image/"
train_label_dir = "data/membrane/train/label/"
test_image_dir = "data/membrane/test"

#transformer = Compose([ToTensor, Normalize([0.5, ], [0.5, ])])
unet_dataloader = Unetdataload(train_image_dir, train_label_dir,batchsize=1).getloader()
# preview = Util.PICPreview(train_image_dir,train_label_dir,2,3)
# preview.view()
cuda = torch.device('cpu')
Unet = Unet().to(cuda)
lr = 0.003
optimizer = optim.Adam(Unet.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()
epochs = 1
for e in range(epochs):
    sum_loss = 0.0
    for _, item in enumerate(unet_dataloader):
        data, label = item
        label = label.type(torch.LongTensor).to(cuda)
        output = Unet(data.to(cuda))
        loss = criterion(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        sum_loss += loss
        if _ % 2 == 1:
            print(sum_loss / 2)
            sum_loss = 0.0

test_pic = image.imread(os.path.join(test_image_dir, "0.png"))
plt.imshow(test_pic)
plt.show()
test_pic = np.expand_dims(np.expand_dims(test_pic, 0), 0)
res = Unet(torch.tensor(test_pic))
res = torch.argmax(res, 1)
res = res.squeeze()
plt.imshow(res)
plt.show()
