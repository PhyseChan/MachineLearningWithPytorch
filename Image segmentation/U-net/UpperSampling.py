import numpy as np
import torch as tr
import torch.nn as nn
import torch.optim as optimizer
import sys
import os
from matplotlib import image
import matplotlib.pyplot as plt
from torchvision.transforms import functional as tvF

class UpSampling:

    def cat(self, target_image, cat_image):
        target_image_len = target_image.shape[-1]
        #cat_layer_len = cat_image.shape[-1]
        croppedimg = tvF.center_crop(cat_image, target_image_len)
        processed_image = tr.cat([croppedimg, target_image], dim=1)
        return processed_image

    def __call__(self, target_image, cat_image):
        processed_image = self.cat(target_image, cat_image)
        return processed_image
