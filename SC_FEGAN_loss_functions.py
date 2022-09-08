import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.applications.vgg16 import VGG16
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model

pool_layers = ["block1_pool", "block2_pool", "block3_pool"]


def get_vgg16_layers(layers_names):
    vgg = VGG16(include_top=False, input_shape=(512, 512, 3), weights='imagenet')
    vgg.trainable = False
    outputs = [vgg.get_layer(name).output for name in layers_names]
    model = Model(inputs=vgg.input, outputs=outputs)
    return model


def compute_content_cost(content_output, generated_output):
    m, n_H, n_W, n_C = content_output.get_shape().as_list()
    a_C_unrolled = tf.reshape(content_output, shape=[m, n_H * n_W, n_C])
    a_G_unrolled = tf.reshape(generated_output, shape=[m, n_H * n_W, n_C])
    J_content = tf.reduce_sum(tf.square(a_C_unrolled - a_G_unrolled)) / (4.0 * n_H * n_W * n_C)
    return J_content

def total_variation_loss(complete_image):
    
    x_var = complete_image[:, :, 1:, :] - complete_image[:, :, :-1, :]
    y_var = complete_image[:, 1:, :, :] - complete_image[:, :-1, :, :]
    return x_var, y_var

def gram_matrix(tensor):
    """tensor shape is [batch,channels,width*height]"""
    """ output shape is [batch,channels ,channels] """
    x = tf.transpose(tensor, (0, 2, 1))
    gram_a = tf.matmul(tensor, x)
    return gram_a


def compute_style_layer(i_out, i_true):
    batch_size, n_h, n_w, n_c = i_true.get_shape().as_list()
    print(batch_size, n_h, n_w, n_c)
    # reshape tensor to [batch_size,width*height,channels]
    x = tf.reshape(i_out, shape=[batch_size, n_h * n_w, n_c])
    x = tf.transpose(x, (0, 2, 1))
    print(x.shape)
    # x = [batch_size,channels,width*height]

    # reshape tensor to [batch_size,width*height,channels]
    y = tf.reshape(i_true, shape=[batch_size, n_h * n_w, n_c])
    y = tf.transpose(y, (0, 2, 1))
    # y = [batch_size,channels,width*height]

    output_1 = gram_matrix(x)
    output_2 = gram_matrix(y)
    print(output_1.shape)

    output_style_layer = (1 / (n_c * n_c)) * tf.reduce_sum(tf.square(output_1 - output_2) / (n_h * n_w * n_c))
    return output_style_layer


def style_loss(gen_output_image, ground_truth_image):
    l_style_gen = 0
    for i in range(len(ground_truth_image)):
        loss = compute_style_layer(gen_output_image[i], ground_truth_image[i])
        l_style_gen += loss
    return l_style_gen


def perceptual_loss(gen_output_image, ground_truth_image):
    term_1 = 0
    for i in range(len(ground_truth_image)):
        _, n_h, n_w, n_c = ground_truth_image[i].get_shape().as_list()
        loss = (gen_output_image[i] - ground_truth_image[i]) / (n_h * n_w * n_c)
        term_1 += loss
    return term_1


def per_pixel_loss(gen_output_image, ground_truth_image, mask):
    term_1 = (1 / 512 * 512 * 3) * (mask * (gen_output_image - ground_truth_image))
    term_2 = (1 / 512 * 512 * 3) * ((1 - mask) * (gen_output_image - ground_truth_image))
    loss = term_1 + 2.0 * term_2
    return loss



def gsn_loss(complete_image):
    gen_loss = - tf.reduce_mean(complete_image)
    return gen_loss





def discriminator_loss(ground_truth_image,complete_image):
    dis_loss = tf.reduce_mean(tf.maximum(1 - ground_truth_image, 0)) + \
               tf.reduce_mean(tf.maximum(1 + complete_image, 0))
    return  dis_loss


