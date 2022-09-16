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
