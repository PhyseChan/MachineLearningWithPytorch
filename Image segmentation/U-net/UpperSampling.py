import numpy as np
import torch as tr
import torch.nn as nn
import torch.optim as optimizer
import sys
import os
from matplotlib import image
import matplotlib.pyplot as plt
from torchvision.transforms import CenterCrop


class UpSampling():
    def __init__(self, transpose_cnn_layer):
        self.transpose_cnn_layer = transpose_cnn_layer

    def transpose(self, target_image):
        transposed_cnn_image = self.transpose_cnn_layer(target_image)
        return transposed_cnn_image

    def cat(self, cat_image, target_image):
        transposed_cnn_image = self.transpose(target_image)
        transposed_cnn_image_len = transposed_cnn_image.shape[-1]
        cat_layer_len = cat_image.shape[-1]
        crop = CenterCrop((transposed_cnn_image_len - cat_layer_len) / 2)
        processed_image = tr.cat([crop(cat_image), self.target_image], dim=1)
        return processed_image

    def __call__(self, target_image, cat_image):
        processed_image = cat_image(target_image, cat_image)
        return processed_image
