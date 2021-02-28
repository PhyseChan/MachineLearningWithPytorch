import os
import matplotlib.image as image
import matplotlib.pyplot as plt
file_path = os.path.join("data/membrane/train/label","0.png")
im = image.imread(file_path)
plt.imshow(im)
