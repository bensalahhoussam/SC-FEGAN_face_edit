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


vgg_extractor = get_vgg16_layers(pool_layers)


def per_pixel_loss(gen_output, ground_truth, mask, alpha):
    _, n_h, n_w, n_c = ground_truth.get_shape().as_list()

    term_1 = tf.reduce_sum(mask * (gen_output - ground_truth)) / (n_h * n_w * n_c)
    term_2 = tf.reduce_sum((1. - mask) * (gen_output - ground_truth)) / (n_h * n_w * n_c)

    loss = term_1 + alpha * term_2
    return loss


def perceptual_loss(complete_image, ground_truth):
    features_ground_truth = vgg_extractor(ground_truth)
    features_output_image = vgg_extractor(complete_image)

    assert len(features_ground_truth) == 3
    assert len(features_output_image) == 3

    term_1 = 0.
    for i in range(len(features_ground_truth)):
        _, n_h, n_w, n_c = features_ground_truth[i].get_shape().as_list()
        loss = tf.reduce_sum(features_output_image[i] - features_ground_truth[i]) / (n_h * n_w * n_c)
        term_1 += loss
    return term_1


def gram_matrix(tensor):
    """tensor shape is [batch,channels,width*height]"""
    """ output shape is [batch,channels ,channels] """
    assert len(tensor.get_shape().as_list()) == 3
    x = tf.transpose(tensor, (0, 2, 1))
    gram_a = tf.matmul(tensor, x)
    return gram_a


def compute_style_layer(gen_output_image, ground_truth_image):
    assert len(gen_output_image.shape) == 4
    assert len(ground_truth_image.shape) == 4

    batch_size, n_h, n_w, n_c = ground_truth_image.get_shape().as_list()
    # reshape tensor to [batch_size,width*height,channels]
    x = tf.reshape(gen_output_image, shape=[batch_size, n_h * n_w, n_c])
    x = tf.transpose(x, (0, 2, 1))
    # x = [batch_size,channels,width*height]

    # reshape tensor to [batch_size,width*height,channels]
    y = tf.reshape(ground_truth_image, shape=[batch_size, n_h * n_w, n_c])
    y = tf.transpose(y, (0, 2, 1))
    # y = [batch_size,channels,width*height]

    output_1 = gram_matrix(x)
    output_2 = gram_matrix(y)

    output_style_layer = tf.reduce_sum(output_1 - output_2) / (n_c * n_c)
    return output_style_layer


def style_loss(gen_output_image, ground_truth):
    features_ground_truth = vgg_extractor(ground_truth)
    features_output_image = vgg_extractor(gen_output_image)

    assert len(features_ground_truth) == 3
    assert len(features_output_image) == 3

    l_style_gen = 0.
    for i in range(len(features_ground_truth)):
        loss = compute_style_layer(features_output_image[i], features_ground_truth[i])
        l_style_gen += loss
    return l_style_gen


def total_variation_loss(complete_image, mask):
    completed = tf.multiply(complete_image, mask)
    zero = tf.constant(0, dtype=tf.float32)
    where = tf.not_equal(completed, zero)
    region = tf.where(where)

    x_var = tf.reduce_sum(
        [tf.reduce_sum(complete_image[:, i + 1, j, :] - complete_image[:, i, j, :]) for i in region[1] for j in
         region[2]])
    x_var = x_var / (512 * 512 * 3)
    y_var = tf.reduce_sum(
        [tf.reduce_sum(complete_image[:, i, j + 1, :] - complete_image[:, i, j, :]) for i in region[1] for j in
         region[2]])
    y_var = y_var / (512 * 512 * 3)
    loss = x_var + y_var
    return loss


def overall_loss(output_gen, ground_truth, complete_image, mask):
    loss = per_pixel_loss(output_gen, ground_truth, mask, alpha=6.)
    perc_loss_1 = perceptual_loss(complete_image, ground_truth)
    perc_loss_2 = perceptual_loss(output_gen, ground_truth)
    style_1 = style_loss(output_gen, ground_truth)
    style_2 = style_loss(complete_image, ground_truth)
    total_variation = total_variation_loss(complete_image, mask)
    total_loss = loss + (0.05 * (perc_loss_1 + perc_loss_2)) + (120 * (style_1 + style_2)) + (0.1 * total_variation)
    return total_loss


def total_generator_loss(output_gen, ground_truth, complete_image, mask, dis_fake, dis_real):
    loss = overall_loss(output_gen, ground_truth, complete_image, mask) + (0.001 * gsn_loss(dis_fake)) + \
           (0.001 * add_term_loss(dis_real))
    return loss


def gsn_loss(dis_fake):
    gen_loss = - tf.reduce_mean(dis_fake)
    return gen_loss


def add_term_loss(dis_real):
    return tf.square(tf.reduce_mean(dis_real))


def gan_hinge_loss(dis_real, dis_fake):
    dis_loss = discriminator_loss(dis_real, dis_fake)
    return dis_loss

def total_dis_loss(dis_real,dis_fake,model,ground_truth,complete_image,mask):
    loss=gan_hinge_loss(dis_real,dis_fake)+10.*gradient_penalty(model,ground_truth,complete_image,mask)
    return loss




def discriminator_loss(dis_real, dis_fake):
    hinge_pos= tf.reduce_mean(1 - dis_real)
    hinge_neg = tf.reduce_mean(1 + dis_fake)
    return tf.add(hinge_pos,hinge_neg)


def gradient_penalty(model, ground_truth, complete_image, mask):
    batch, w, h, c = ground_truth.get_shape().as_list()

    alpha = tf.random.normal([batch, 1, 1, 1], 0.0, 1.0)
    diff = complete_image - ground_truth
    interpolated = (complete_image + alpha * diff) * mask

    with tf.GradientTape() as gp_tape:
        gp_tape.watch(interpolated)
        pred = model(interpolated, training=True)
    grads = gp_tape.gradient(pred, [interpolated])[0]
    norm = tf.sqrt(tf.reduce_sum(tf.square(grads)))
    gp = tf.reduce_mean((norm - 1.0) ** 2)
    return gp
