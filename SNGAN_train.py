import os
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_io as tfio
from keras.layers import Input
from build_SNGAN_generator import Generator
from build_SNGAN_discriminator import Discriminator

path = "D://Deep_Learning_projects/new_projects/computer_vision/project_3/dataset/"


class Data_Preparation:
    def __init__(self, folder_path):
        self.path = folder_path
        self.total_images, self.total_sketch, self.total_color, self.total_mask, self.total_noise = \
            self.data_load()
        self.total_batch = self.data_batch()
        self.incomplete_image = self.total_batch[:][..., 0:3]
        self.sketch = self.total_batch[:][..., 3:4]
        self.color = self.total_batch[:][..., 4:7]
        self.mask = self.total_batch[:][..., 7:8]
        self.noise = self.total_batch[:][..., 8:9]
        self.data = [self.incomplete_image, self.sketch, self.color, self.mask, self.noise]

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
        for i in range(len(self.total_sketch[0:1])):
            pic = tf.image.decode_jpeg(tf.io.read_file(self.total_images[i]), channels=3)
            pic = tfio.experimental.color.bgr_to_rgb(pic)
            pic = tf.cast(pic, dtype=tf.float32) / 127.5 - 1.

            sketch = tf.image.decode_jpeg(tf.io.read_file(self.total_sketch[i]), channels=1)
            sketch = tf.cast(sketch, dtype=tf.float32) / 255.

            color = tf.image.decode_jpeg(tf.io.read_file(self.total_color[i]), channels=3)
            color = tfio.experimental.color.bgr_to_rgb(color)
            color = tf.cast(color, dtype=tf.float32) / 255.

            mask = tf.image.decode_jpeg(tf.io.read_file(self.total_mask[i]),channels=1)
            mask = tf.cast(mask, dtype=tf.float32) / 255.


            noise = tf.image.decode_jpeg(tf.io.read_file(self.total_noise[i]),channels=1)
            noise = tf.cast(noise, dtype=tf.float32) / 255.

            batch_input = tf.concat([pic, sketch, color, mask, noise], axis=-1)
            total_batch.append(batch_input)
        total_batch = tf.stack(total_batch,axis=0)
        return total_batch


    def complete_image(self,output_gen):
        image = self.incomplete_image + (self.mask * output_gen)
        return image


data = Data_Preparation(path)
data_1 = data.data


def model():
    input_gen = Input(shape=(512, 512, 9))
    input_dis = Input(shape=(512, 512, 5))
    model_1 = Generator()
    model_2 = Discriminator()
    model_generator = model_1.call(input_gen)
    model_discriminator = model_2.call(input_dis)
    return model_generator, model_discriminator


model_generator, model_discriminator = model()
input_gen = data.total_batch
input_dis = tf.concat([data_1[0], data_1[1], data_1[3]], axis=-1)
output_gen = model_generator(input_gen)
output_dis = model_discriminator(input_dis)

print(f"batch_input shape : {input_gen.shape}")
print(f"generator_output shape : {output_gen.shape}")
print(f"discriminator input shape : {input_dis.shape}")
print(f"discriminator output shape : {output_dis.shape}")

print("*" * 50)
data_1 = data.data
print(f"incomplete_images shape : {data_1[0].shape}")
print(f"sketch shape : {data_1[1].shape}")
print(f"color shape : {data_1[2].shape}")
print(f"mask shape : {data_1[3].shape}")
print(f"noise shape : {data_1[4].shape}")

batch_input shape : (10, 512, 512, 9)
generator_output shape : (10, 512, 512, 3)
discriminator input shape : (10, 512, 512, 5)
discriminator output shape : (10, 16, 16, 256)
**************************************************
incomplete_images shape : (10, 512, 512, 3)
sketch shape : (10, 512, 512, 1)
color shape : (10, 512, 512, 3)
mask shape : (10, 512, 512, 1)
noise shape : (10, 512, 512, 1)
