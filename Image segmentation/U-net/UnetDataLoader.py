import numpy as np
import torch as tr
import torch.nn as nn
import torch.optim as optimizer
import sys
import os
from matplotlib import image
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import CenterCrop, Compose


class Unetdataload():
    def __init__(self, train_image_dir, train_label_dir):
        self.train_image_fileName_list = os.listdir(train_image_dir)
        self.train_image_folder_path = train_image_dir
        self.train_label_folder_path = train_label_dir

    def preview(self):
        preview_images_list = []
        preview_label_list = []
        preview_number = 3
        for i in range(preview_number):
            image_file_path = os.path.join(self.train_image_folder_path, self.train_image_fileName_list[i])
            label_file_path = os.path.join(self.train_label_folder_path, self.train_image_fileName_list[i])
            preview_image_data = image.imread(image_file_path)
            preview_images_list.append(preview_image_data)
            preview_label_data = image.imread(label_file_path)
            preview_label_list.append(preview_label_data)
        for i in range(preview_number):
            plt.subplot(2, preview_number, i + 1)
            plt.imshow(preview_images_list[i], cmap="gray")
            plt.subplot(2, preview_number, i + 1 + preview_number)
            plt.imshow(preview_label_list[i], cmap="gray")
        plt.show()

    def getdataset(self):
        dataset = PicDataset(self.train_image_folder_path,self.train_label_folder_path,self.train_image_fileName_list)
        return dataset

class PicDataset(Dataset):
    def __init__(self, data_path, label_path, file_name_list, transform=None):

        self.data_path = data_path
        self.label_path = label_path
        self.file_name_list = file_name_list
        self.transform = transform

    def __len__(self):
        return len(self.file_name_list)

    def __getitem__(self, idx):
        data_file_path = os.path.join(self.data_path, self.file_name_list[idx])
        label_file_path = os.path.join(self.data_path, self.file_name_list[idx])
        data = image.imread(data_file_path)
        label = image.imread(label_file_path)
        if self.transform is not None:
            data = tr.autograd.variable(data)
            label = tr.autograd.variable(label)
            label = self.transform(label)
        else:
            data = tr.autograd.variable(data)
            label = tr.autograd.variable(label)
        return (data, label)
