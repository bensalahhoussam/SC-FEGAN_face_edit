import os
import cv2 as cv
import numpy as np
import tensorflow as tf
import tensorflow_io as tfio
from keras.layers import Input
from build_SNGAN_discriminator import Discriminator
from build_SNGAN_generator import Generator
from SC_FEGAN_data_preparation import Data_Preparation, model
from SC_FEGAN_loss_functions import generator_loss_function, per_pixel_loss, perceptual_loss, style_loss, \
    total_variation_loss

path = "D://Deep_Learning_projects/new_projects/computer_vision/project_3/dataset/"

model_generator, model_discriminator = model()

data = Data_Preparation(path)

input_gen = data.total_input
ground_truth = data.ground_truth
batch_data = data.batch_data

incomplete_image = batch_data[0]
sketch = batch_data[1]
color = batch_data[2]
mask = batch_data[3]
noise = batch_data[4]

output_gen = model_generator(input_gen)
complete_image = data.complete_image(output_gen)

gen_loss=generator_loss_function(output_gen,ground_truth,complete_image,mask)
















"""print("*" * 50)
print(f"ground_truth shape : {ground_truth.shape}")
print(f"batch_input shape : {input_gen.shape}")
print(f"generator_output shape : {output_gen.shape}")
print(f"discriminator input shape : {input_dis.shape}")
print(f"discriminator output shape : {output_dis.shape}")
print("*" * 50)
print(f"incomplete_images shape : {data_1[0].shape}")
print(f"sketch shape : {data_1[1].shape}")
print(f"color shape : {data_1[2].shape}")
print(f"mask shape : {data_1[3].shape}")
print(f"noise shape : {data_1[4].shape}")"""
