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


def per_pixel_loss(gen_output_image, ground_truth_image, mask, alpha):
    _, n_h, n_w, n_c = ground_truth_image.get_shape().as_list()
    term_1 = tf.reduce_sum(mask * (gen_output_image - ground_truth_image)) / (n_h * n_w, n_c)
    term_2 = tf.reduce_sum((1 - mask) * (gen_output_image - ground_truth_image)) / (n_h * n_w, n_c)
    loss = term_1 + alpha * term_2
    return loss

def perceptual_loss(incomplete_image, ground_truth):
    features_ground_truth = vgg_extractor(ground_truth)
    features_output_image = vgg_extractor(incomplete_image)

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
    print("shape",batch_size, n_h, n_w, n_c)
    # reshape tensor to [batch_size,width*height,channels]
    x = tf.reshape(gen_output_image, shape=[batch_size, n_h * n_w, n_c])
    x = tf.transpose(x, (0, 2, 1))
    print(x.shape)
    # x = [batch_size,channels,width*height]

    # reshape tensor to [batch_size,width*height,channels]
    y = tf.reshape(ground_truth_image, shape=[batch_size, n_h * n_w, n_c])
    y = tf.transpose(y, (0, 2, 1))
    print(y.shape)
    # y = [batch_size,channels,width*height]

    output_1 = gram_matrix(x)
    output_2 = gram_matrix(y)
    print(output_1.shape,output_2.shape)

    output_style_layer = tf.reduce_sum(output_1 - output_2)/(n_c * n_c)
    return output_style_layer

def style_loss(gen_output_image):
    features_ground_truth = vgg_extractor(ground_truth)
    features_output_image = vgg_extractor(gen_output_image)

    assert len(features_ground_truth) == 3
    assert len(features_output_image) == 3

    l_style_gen = 0.
    for i in range(len(features_ground_truth)):
        loss = compute_style_layer(features_output_image[i], features_ground_truth[i])
        l_style_gen += loss
    return l_style_gen

def total_variation_loss(complete_image,mask):
    h,w,c=complete_image[1:]
    completed=tf.multiply(complete_image,mask)
    zero = tf.constant(0, dtype=tf.float32)
    where = tf.not_equal(completed, zero)
    region = tf.where(where)

    x_var = tf.reduce_sum([tf.reduce_sum(complete_image[:, i+1, j, :] - complete_image[:, i, j, :]) for i in region[1] for j in region[2]])
    x_var=x_var/(w*h*c)
    y_var = tf.reduce_sum([tf.reduce_sum(complete_image[:, i, j+1, :] - complete_image[:, i, j, :]) for i in region[1] for j in region[2]])
    y_var = y_var / (w*h*c)
    loss=x_var+y_var
    return loss


    return x_var+ y_var

def gsn_loss(complete_image):
    gen_loss = - tf.reduce_mean(complete_image)
    return gen_loss

def add_term_loss(true_image):
    return tf.square(tf.reduce_mean(true_image))



def generator_loss_function(generated_image,ground_truth,incomplete_image,mask):
    gamma=0.05
    betta=0.001
    v=0.1
    epsilon=0.001
    delta=120
    ppxl_loss=per_pixel_loss(generated_image, ground_truth, mask, alpha=2.)
    perc_loss_1=perceptual_loss(incomplete_image, ground_truth)
    perc_loss_2=perceptual_loss(generated_image, ground_truth)
    l_gsn = gsn_loss(incomplete_image)
    sg_loss=style_loss(generated_image)
    sc_loss=style_loss(incomplete_image)
    total_variation=total_variation_loss(incomplete_image,mask)
    dis_gt=add_term_loss(ground_truth)

    overall_loss = ppxl_loss+(gamma*(perc_loss_1+perc_loss_2))+(betta*l_gsn)+(delta*(sg_loss+sc_loss))+\
                   (v*total_variation)+(epsilon*dis_gt)
    return overall_loss


def averaged_samples(real,fake):
    alpha = K.random_uniform((batch_size, 1, 1, 1))
    return (alpha * real) + ((1 - alpha) * fake)



def gp_loss(y_true, y_pred, averaged_samples):
    gradients = K.gradients(y_pred, averaged_samples)[0]
    gradients_sqr = K.square(gradients)
    gradients_sqr_sum = K.sum(gradients_sqr,
                            axis=np.arange(1, len(gradients_sqr.shape)))
    gradient_l2_norm = K.sqrt(gradients_sqr_sum)
    gradient_penalty = K.square(1 - gradient_l2_norm)
    return K.mean(gradient_penalty)



def gan_hinge_loss(fake, real):
    dis_loss = discriminator_loss(real, fake)
    return dis_loss

def discriminator_loss(ground_truth_image, complete_image):
    dis_loss = tf.reduce_mean(tf.maximum(1 - ground_truth_image, 0)) + \
               tf.reduce_mean(tf.maximum(1 + complete_image, 0))
    return dis_loss
