import torchvision.datasets.mnist as mnist
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, ToTensor, Normalize

class MnistReader():
    def __init__(self, transformer=None):
        if transformer == None:
            transformer = Compose([ToTensor(), Normalize([0.5],[0.5])])
        self.training_data = mnist.MNIST("../data/", train=True, transform=transformer, download=True)
        self.test_data = mnist.MNIST("../data/", train=False, transform=transformer, download=True)

    def getdataloader(self, batch_size_trian=10, batch_size_test=10):
        training_loader = DataLoader(self.training_data,batch_size_trian,shuffle=True)
        test_loader = DataLoader(self.test_data, batch_size_test)
        return training_loader, test_loader