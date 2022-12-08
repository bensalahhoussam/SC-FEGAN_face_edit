import os
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import tensorflow as tf
import tensorflow_io as tfio
from keras.layers import Input
from build_SNGAN_discriminator import Discriminator
from build_SNGAN_generator import Generator
from SC_FEGAN_data_preparation import Data_Preparation
from SC_FEGAN_loss_functions import  total_generator_loss,total_dis_loss,vgg_extractor
from tqdm import tqdm,trange
from tensorflow.keras.optimizers import Adam
from keras.models import model_from_json
from loss_gpu import total_generator_loss,total_dis_loss
from path_file import args
from time import sleep

def set_global_batch_size(batch_size_per_replica, strategy):
    global_batch_size = batch_size_per_replica * strategy.num_replicas_in_sync
    return global_batch_size
def data_split(data):
    input_gen = data.total_input
    ground_truth = data.ground_truth
    batch_data = data.batch_data

    incomplete_image,sketch ,color ,mask,noise = batch_data[0],batch_data[1],batch_data[2],batch_data[3],\
                                                 batch_data[4]

    return input_gen,ground_truth,incomplete_image,sketch,color,mask,noise
def data_distrution(input):
    data=tf.data.Dataset.from_tensor_slices((input)).batch(global_batch_size ,drop_remainder=True).prefetch(-1).cache()
    return data
def data_distrubuted(path):
    data = Data_Preparation(path)
    input_gen, ground_truth, incomplete_image, sketch, color, mask, noise = data_split(data)

    input_data = data_distrution(input_gen)
    incomplete_data = data_distrution(incomplete_image)
    mask_data = data_distrution(mask)
    real_data = data_distrution(ground_truth)
    sketch_data = data_distrution(sketch)
    color_data = data_distrution(color)
    return  input_data,mask_data,incomplete_data,real_data,sketch_data,color_data
def distribute_datasets(strategy, total_data):
    input_dist_dataset = strategy.experimental_distribute_dataset(total_data[0])
    mask_dist_dataset = strategy.experimental_distribute_dataset(total_data[1])
    incomplete_dist_dataset = strategy.experimental_distribute_dataset(total_data[2])
    real_dist_dataset = strategy.experimental_distribute_dataset(total_data[3])
    sketch_dist_dataset = strategy.experimental_distribute_dataset(total_data[4])
    color_dist_dataset = strategy.experimental_distribute_dataset(total_data[5])

    return input_dist_dataset,mask_dist_dataset,incomplete_dist_dataset,real_dist_dataset,sketch_dist_dataset,color_dist_dataset
def model():
    gen_input = Input(shape=(512, 512, 9))
    dis_input = Input(shape=(512, 512, 8))
    model_1 = Generator()
    model_2 = Discriminator()
    gen_model = model_1.call(gen_input)
    dis_model = model_2.call(dis_input)
    return gen_model, dis_model


mirrored_strategy = tf.distribute.MirroredStrategy()
print(f"Number of devices: {mirrored_strategy.num_replicas_in_sync}")

global_batch_size = set_global_batch_size(args.batch_size_per_replica,mirrored_strategy)

total_data=data_distrubuted(args.training_dataset)

input_dist,mask_dist,incomplete_dist,real_dist,sketch_dist,color_dist=distribute_datasets(mirrored_strategy,total_data)




with mirrored_strategy.scope():
    model_generator,model_discriminator = model()
    generator_optimizer = Adam(learning_rate=1e-4, beta_1=0.1, beta_2=0.999)
    discriminator_optimizer = Adam(learning_rate=1e-4, beta_1=0.1, beta_2=0.999)

with mirrored_strategy.scope():
    def compute_generator_loss(output_gen, ground_truth, complete_image, mask, dis_fake, dis_real):
        gen_loss=total_generator_loss(output_gen, ground_truth, complete_image, mask, dis_fake, dis_real)
        return tf.nn.compute_average_loss(gen_loss,global_batch_size=4)

    def compute_discriminator_loss(model,dis_real,dis_fake,ground_truth,complete_image,mask):
        dis_loss = total_dis_loss(model,dis_real,dis_fake,ground_truth,complete_image,mask)
        return tf.nn.compute_average_loss(dis_loss,global_batch_size=4)

with mirrored_strategy.scope():
    def apply_gradient(input_gen, mask, incomplete_image, ground_truth, sketch, color ):

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            output_gen = model_generator(input_gen)

            complete_image = incomplete_image + (mask * output_gen)

            batch_pos= tf.concat([ground_truth,sketch,color,mask],axis=-1)
            batch_neg = tf.concat([complete_image,sketch,color,mask], axis=-1)

            dis_real = model_discriminator(batch_pos)
            dis_fake = model_discriminator(batch_neg)

            gen_loss = compute_generator_loss(output_gen, ground_truth, complete_image, mask, dis_fake, dis_real)
            dis_loss = compute_discriminator_loss(dis_real,dis_fake,model_discriminator,batch_pos,batch_neg,mask)


        generator_gradients = gen_tape.gradient(gen_loss,model_generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(dis_loss,model_discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(generator_gradients,model_generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(discriminator_gradients,model_discriminator.trainable_variables))

        return gen_loss,dis_loss

with mirrored_strategy.scope():
    @tf.function
    def distributed_train_step(input_gen, mask, incomplete_image, ground_truth, sketch, color):
        per_replica_losses_gen,per_replica_losses_dis= mirrored_strategy.run(apply_gradient, args=(input_gen, mask, incomplete_image, ground_truth, sketch, color,))
        gen_loss=mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses_gen,axis=None)
        dis_loss = mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses_dis,axis=None)
        return gen_loss,dis_loss

def train_fit(epochs):
    for epoch in range(epochs):
        print(f"Start of epoch number : {epoch}")
        total_loss_gen = 0.0
        total_loss_dis = 0.0
        total_batch = 0


        dataset = zip(input_dist, mask_dist, incomplete_dist, real_dist, sketch_dist, color_dist)

        pbar = tqdm(dataset,total=len(list(enumerate(input_dist))), position=0, leave=True,
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}')
        step = 0
        for batch in pbar:

            gen_loss,dis_loss= distributed_train_step(batch[0],batch[1],batch[2],batch[3],batch[4],batch[5])

            total_loss_gen+=gen_loss
            total_loss_dis+=dis_loss
            total_batch+=1

            pbar.set_description(f"Training loss for step_num {step} ,gen_loss:{gen_loss:0.4f},dis_loss:{dis_loss:0.4f}")
            step +=1
            pbar.update()
        total_loss_gen /= total_batch
        total_loss_dis /= total_batch
        tf.print(f'Epoch {epoch}: total gen loss is : {total_loss_gen:.3f}  total dis loss: {total_loss_dis:.3f}')







