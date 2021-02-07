import torch as tr
import os
from matplotlib import image
from torch.utils.data import DataLoader, Dataset
import numpy as np


class Unetdataload():
    def __init__(self, train_image_dir, train_label_dir, batchsize = 4):
        self.train_image_fileName_list = os.listdir(train_image_dir)
        self.train_image_folder_path = train_image_dir
        self.train_label_folder_path = train_label_dir
        self.batchsize = batchsize

    def getloader(self):
        dataset = PicDataset(self.train_image_folder_path,self.train_label_folder_path,self.train_image_fileName_list)
        dataloaer = DataLoader(dataset, self.batchsize, shuffle= True)
        return dataloaer

class PicDataset(Dataset):
    def __init__(self, data_path, label_path, file_name_list):

        self.data_path = data_path
        self.label_path = label_path
        self.file_name_list = file_name_list

    def __len__(self):
        return len(self.file_name_list)

    def __getitem__(self, idx):
        data_file_path = os.path.join(self.data_path, self.file_name_list[idx])
        label_file_path = os.path.join(self.data_path, self.file_name_list[idx])
        data = image.imread(data_file_path)
        label = image.imread(label_file_path)
        data = np.expand_dims(data,axis=0)
        return data, label
