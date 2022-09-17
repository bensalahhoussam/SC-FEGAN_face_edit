import os
import cv2 as cv
import numpy as np
import tensorflow as tf
import tensorflow_io as tfio
from keras.layers import Input
from build_SNGAN_discriminator import Discriminator
from build_SNGAN_generator import Generator
from SC_FEGAN_data_preparation import Data_Preparation, model,load_data
from SC_FEGAN_loss_functions import  overall_loss,gsn_loss,gan_hinge_loss,add_term_loss,total_generator_loss


path = "D://Deep_Learning_projects/new_projects/computer_vision/project_3/dataset/"
model_generator, model_discriminator = model()

data = Data_Preparation(path)

def data_split(data):
    input_gen = data.total_input
    ground_truth = data.ground_truth
    batch_data = data.batch_data

    incomplete_image,sketch ,color ,mask,noise = batch_data[0],batch_data[1],batch_data[2],batch_data[3],\
                                                 batch_data[4]

    return input_gen,ground_truth,incomplete_image,sketch,color,mask,noise

input_gen,ground_truth,incomplete_image,sketch,color,mask,noise=data_split(data)

output_gen = model_generator(input_gen)
complete_image = data.complete_image(output_gen)

batch_neg=tf.concat([complete_image,mask,sketch],axis=-1)
batch_pos=tf.concat([ground_truth,mask,sketch],axis=-1)
batch_global=tf.concat([batch_pos,batch_neg],axis=0)
dis_real_fake=model_discriminator(batch_global)
dis_real,dis_fake = tf.split(dis_real_fake,2)

total_loss=total_generator_loss(output_gen,ground_truth,complete_image,mask,dis_fake,dis_real)

dis_loss=gan_hinge_loss(dis_real,dis_fake)
print(dis_loss)



