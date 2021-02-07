import numpy as np
import torch as tr

class UpSampling:

    def cat(self, target_image, cat_image):
        target_image_len = target_image.shape[-1]
        cat_image_len = cat_image.shape[-1]
        croplen = int((cat_image_len - target_image_len)/2)
        croppedimg = cat_image[:, :, croplen:croplen+target_image_len, croplen:croplen+target_image_len]
        processed_image = tr.cat([croppedimg, target_image], dim=1)
        return processed_image

    def __call__(self, target_image, cat_image):
        processed_image = self.cat(target_image, cat_image)
        return processed_image
