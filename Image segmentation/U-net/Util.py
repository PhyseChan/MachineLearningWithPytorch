import matplotlib.image as image
import matplotlib.pyplot as plt
import os

class PICPreview():
    def __init__(self, data_path,label_path, witdh, length):
        self.n_images = witdh * length
        self.width = witdh
        self.length = length
        self.label_path = label_path
        self.data_path = data_path
        self.file_name_list = os.listdir(data_path)

    def view(self):
        preview_images_list = []
        preview_label_list = []
        for i in range(self.n_images):
            image_file_path = os.path.join(self.data_path, self.file_name_list[i])
            label_file_path = os.path.join(self.label_path, self.file_name_list[i])
            preview_image_data = image.imread(image_file_path)
            preview_images_list.append(preview_image_data)
            preview_label_data = image.imread(label_file_path)
            preview_label_list.append(preview_label_data)
        for i in range(self.n_images):
            plt.subplot(2*self.width,self.length , i + 1)
            plt.imshow(preview_images_list[i], cmap="gray")
            plt.subplot(2*self.width, self.length, i + 1 + self.n_images)
            plt.imshow(preview_label_list[i], cmap="gray")
        plt.show()


