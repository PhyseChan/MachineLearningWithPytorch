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

train_image_dir = "data/membrane/train/image/"
train_label_dir = "data/membrane/train/label/"

test_image_dir = "data/membrane/test/image"
test_label_dir = "data/membrane/test/label"

unet_dataloader = Unetdataload(train_image_dir, train_label_dir, batchsize=1).getloader()

# preview = Util.PICPreview(train_image_dir,train_label_dir,2,3)
# preview.view()
cuda = torch.device('cpu')
Unet = Unet().to(cuda)
lr = 0.003
optimizer = optim.Adam(Unet.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()
epochs = 20
for e in range(epochs):
    sum_loss = 0.0
    for _, item in enumerate(unet_dataloader):
        data, label = item
        label = label.to(cuda)
        output = Unet(data.to(cuda))
        loss = criterion(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        sum_loss += loss
        if _ % 10 == 9:
            print(sum_loss / 10)
            sum_loss = 0.0

unet_dataloader_test = Unetdataload(train_image_dir, train_label_dir, batchsize=1).getloader()
Unet.cpu()
Unet.eval()
with torch.no_grad():
    sum_loss = 0.0
    images_list = []
    labels_list = []
    for _, item in enumerate(unet_dataloader_test):
        data, label = item
        output = Unet(data)
        sum_loss += criterion(output, label)
        output_images = torch.argmax(output, 1)
        images_list.append(output_images.squeeze().numpy())
        labels_list.append(label.squeeze().numpy())

test_pic = image.imread(os.path.join(test_image_dir, "0.png"))
plt.imshow(test_pic)
plt.show()
test_pic = np.expand_dims(np.expand_dims(test_pic, 0), 0)
res = Unet(torch.tensor(test_pic))
res = torch.argmax(res, 1)
res = res.squeeze()
plt.imshow(res)
test_image_list = zip(images_list[:3], labels_list[:3])
i = 0
for _, item in enumerate(test_image_list):
    image, label = item
    plt.subplot(3, 2, _*2+1)
    plt.imshow(image)
    plt.subplot(3, 2, _*2+2)
    plt.imshow(label)
plt.show()
print(sum_loss / (_ + 1))

