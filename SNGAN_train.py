import os
import cv2 as cv
import numpy as np
import tensorflow as tf
import tensorflow_io as tfio
from keras.layers import Input
from build_SNGAN_discriminator import Discriminator
from build_SNGAN_generator import Generator
from SC_FEGAN_data_preparation import Data_Preparation,load_data
from SC_FEGAN_loss_functions import  total_generator_loss,discriminator_loss,\
    gan_hinge_loss,gradient_penalty

from tensorflow.keras.optimizers import Adam
path = "D://Deep_Learning_projects/new_projects/computer_vision/project_3/dataset/"

def model():
    gen_input = Input(shape=(512, 512, 9))
    dis_input = Input(shape=(512, 512, 5))
    model_1 = Generator()
    model_2 = Discriminator()
    gen_model = model_1.call(gen_input)
    dis_model = model_2.call(dis_input)
    return gen_model, dis_model
data = Data_Preparation(path)

def data_split(data):
    input_gen = data.total_input
    ground_truth = data.ground_truth
    batch_data = data.batch_data

    incomplete_image,sketch ,color ,mask,noise = batch_data[0],batch_data[1],batch_data[2],batch_data[3],\
                                                 batch_data[4]

    return input_gen,ground_truth,incomplete_image,sketch,color,mask,noise

def data_distrution(input):
    data=tf.data.Dataset.from_tensor_slices((input)).batch(1)
    return data


model_generator,model_discriminator = model()
edges_detection = load_data()


generator_optimizer = Adam(1e-4, beta_1=0.1, beta_2=0.999)
discriminator_optimizer = Adam(1e-4, beta_1=0.1, beta_2=0.999)

input_gen,ground_truth,incomplete_image,sketch,color,mask,noise=data_split(data)


input_data = data_distrution(input_gen)
incomplete_data = data_distrution(incomplete_image)
mask_data = data_distrution(mask)
real_data = data_distrution(ground_truth)
sketch_data = data_distrution(sketch)


def train_one_batch(input_gen,mask,incomplete_image,ground_truth,sketch):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

        output_gen = model_generator(input_gen)

        complete_image = incomplete_image + (mask * gen_output)

        true_mask = tf.ones_like(complete_image)[...,0:1]
        true_sketch = edges_detection.edges_detection(ground_truth)


        batch_pos= tf.concat([ground_truth,true_sketch,true_mask],axis=-1)
        batch_neg = tf.concat([complete_image,sketch,mask], axis=-1)

        dis_real = model_discriminator(batch_pos)
        dis_fake = model_discriminator(batch_neg)

        gen_loss = total_generator_loss(output_gen, ground_truth, complete_image, mask, dis_fake, dis_real)

        dis_loss= gan_hinge_loss(dis_real=dis_real,dis_fake=dis_fake)

        gp = gradient_penalty(model_discriminator,batch_pos,batch_neg,mask)

        total_dis_loss = dis_loss +(10.*gp)


        generator_gradients = gen_tape.gradient(gen_loss,
                                                model_generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(total_dis_loss,
                                                     model_discriminator.trainable_variables)


def train_one_epoch(epochs=1):
    for i in range(epochs):
        for true,input,incomplete,mask,sketch in zip(real_data,input_data,incomplete_data
                                                     ,mask_data,sketch_data):
            train_one_batch(input,mask,incomplete,true,sketc)
