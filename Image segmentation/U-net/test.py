from model import SimpleUnet
from torch.utils.data import DataLoader
from UnetDataLoader import Unetdataload, PicDataset
Unet = SimpleUnet()
train_image_dir = "data/membrane/train/image/"
train_label_dir = "data/membrane/train/label/"

unetdata = Unetdataload(train_image_dir, train_label_dir)
Unet_dataset = unetdata.getdataset()
Unet_dataloader = DataLoader(Unet_dataset, batch_size=4, shuffle=True,)
print()