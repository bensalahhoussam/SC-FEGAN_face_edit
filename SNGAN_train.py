import os
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.layers import Input


path = "D://Deep_Learning_projects/new_projects/computer_vision/project_3/dataset/"


class Data_Preparation:
    def __init__(self, folder_path):
        self.path = folder_path
        self.total_images, self.total_sketch, self.total_color, self.total_mask, self.total_noise = \
            self.data_load()
        self.total_batch = self.data_batch()
        self.incomplete_image=self.total_batch[:][...,0:3]
        self.sketch=self.total_batch[:][...,3:4]
        self.color=self.total_batch[:][...,4:7]
        self.mask=self.total_batch[:][...,7:8]
        self.noise=self.total_batch[:][...,8:9]
        self.data=[self.incomplete_image,self.sketch,self.color,self.mask,self.noise]


    def data_load(self, ):
        images = []
        colors = []
        edges = []
        masks = []
        noises = []

        data_folder = [name for name in os.listdir(self.path)]

        images_color = [img for img in os.listdir(self.path + "/" + data_folder[0])]
        images_input = [img for img in os.listdir(self.path + "/" + data_folder[1])]

        images_mask = [img for img in os.listdir(self.path + "/" + data_folder[2])]

        images_noise = [img for img in os.listdir(self.path + "/" + data_folder[3])]

        images_sketch = [img for img in os.listdir(self.path + "/" + data_folder[4])]

        for i in range(len(images_color)):
            image_path, sketch_path, color_path, mask_path, noise_path = images_input[i], images_sketch[i], \
                                                                         images_color[i], \
                                                                         images_mask[i], images_noise[i]

            images.append(self.path + data_folder[1] + "/" + image_path)
            edges.append(self.path + data_folder[4] + "/" + sketch_path)
            colors.append(self.path + data_folder[0] + "/" + color_path)
            masks.append(self.path + data_folder[2] + "/" + mask_path)
            noises.append(self.path + data_folder[3] + "/" + noise_path)
        return images, edges, colors, masks, noises

    def data_batch(self, ):
        total_batch = []
        for i in range(len(self.total_sketch[0:50])):
            image = cv.cvtColor(cv.imread(self.total_images[i]),cv.COLOR_BGR2RGB)
            sketch = cv.imread(self.total_sketch[i])
            sketch = sketch[..., 0:1]
            color = cv.cvtColor(cv.imread(self.total_color[i]),cv.COLOR_BGR2RGB)
            mask = cv.imread(self.total_mask[i])
            mask = mask[..., 0:1]
            noise = cv.imread(self.total_noise[i])
            noise = noise[..., 0:1]
            batch_input = np.concatenate([image, sketch, color, mask, noise], axis=-1)
            total_batch.append(batch_input)
        total_batch = np.array(total_batch)
        return total_batch

data = Data_Preparation(path)


input_batch=data.total_batch
print(f"total_batch shape : {input_batch.shape}")
data_1=data.data
print(f"incomplete_images shape : {data_1[0].shape}")
print(f"sketch shape : {data_1[1].shape}")
print(f"color shape : {data_1[2].shape}")
print(f"mask shape : {data_1[3].shape}")
print(f"noise shape : {data_1[4].shape}")




total_batch shape : (50, 512, 512, 9)
incomplete_images shape : (50, 512, 512, 3)
sketch shape : (50, 512, 512, 1)
color shape : (50, 512, 512, 3)
mask shape : (50, 512, 512, 1)
noise shape : (50, 512, 512, 1)




