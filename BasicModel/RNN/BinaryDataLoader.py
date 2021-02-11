from torch.utils.data import Dataset, DataLoader
import torch as tr

class BinaryDataSet(Dataset):
    def __init__(self, data, cat):
        self.data = data
        self.cat = cat

    def __getitem__(self, index):
        item = self.data[index]
        data = tr.tensor(item[:2], dtype = tr.float)
        # label = tr.zeros(self.cat)
        # label[item[2]] = 1
        label = tr.tensor(item[2], dtype = tr.long)
        return data, label

    def __len__(self):
        return len(self.data)
