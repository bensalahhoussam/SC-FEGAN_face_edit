import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation, Input, Conv2D, LeakyReLU, Multiply, Conv2DTranspose, Concatenate
from tensorflow.keras.utils import plot_model


def Gated_Conv_2D(x_layer, filters, lrn=True, dilation_rate=1, strides=2, num_block=1, output_layer=False):
    x_feature = Conv2D(filters=filters, kernel_size=3, padding="same", strides=strides, dilation_rate=dilation_rate,
                       name="conv_1_" + str(num_block))(x_layer)
    if lrn is True:
        x_feature = tf.nn.local_response_normalization(x_feature, depth_radius=5, bias=1, alpha=1, beta=0.5,
                                                       name="LRN_1_" + str(num_block))

    if output_layer is False:
        x_feature = LeakyReLU(alpha=0.1)(x_feature)
    else:
        x_feature = Activation("tanh")(x_feature)

    x_gating = Conv2D(filters=filters, kernel_size=1, padding="same", strides=strides,
                      dilation_rate=dilation_rate, name="gated_conv_1_" + str(num_block))(x_layer)
    x_gating = Activation("sigmoid", name="sigmoid_1_" + str(num_block))(x_gating)
    output = Multiply(name="multiplication_layer_1_" + str(num_block))([x_gating, x_feature])

    x_feature = Conv2D(filters=filters, kernel_size=3, padding="same", dilation_rate=dilation_rate,
                       name="conv_2_" + str(num_block))(output)
    if lrn is True:
        x_feature = tf.nn.local_response_normalization(x_feature, depth_radius=5, bias=1, alpha=1, beta=0.5,
                                                       name="LRN_2_" + str(num_block))

    if output_layer is False:
        x_feature = LeakyReLU(alpha=0.1)(x_feature)
    else:
        x_feature = Activation("tanh")(x_feature)

    x_gating = Conv2D(filters=filters, kernel_size=1, padding="same", dilation_rate=dilation_rate,
                      name="gated_conv_2_" + str(num_block))(output)
    x_gating = Activation("sigmoid", name="sigmoid_2_" + str(num_block))(x_gating)
    output = Multiply(name="multiplication_layer_2_" + str(num_block))([x_gating, x_feature])

    x_feature = Conv2D(filters=filters, kernel_size=3, padding="same", name="conv_3_" + str(num_block))(output)
    if lrn is True:
        x_feature = tf.nn.local_response_normalization(x_feature, depth_radius=5, bias=1, alpha=1, beta=0.5,
                                                       name="LRN_3_" + str(num_block))

    if output_layer is False:
        x_feature = LeakyReLU(alpha=0.1)(x_feature)
    else:
        x_feature = Activation("tanh")(x_feature)

    return x_feature
def encoder(x_layer):
    skip_1 = x_layer
    x = Gated_Conv_2D(x_layer, filters=64, lrn=False, dilation_rate=1, num_block=1)
    skip_2 = x
    x = Gated_Conv_2D(x, filters=128, lrn=True, dilation_rate=1, num_block=2)
    skip_3 = x
    x = Gated_Conv_2D(x, filters=256, lrn=True, dilation_rate=1, num_block=3)
    skip_4 = x
    x = Gated_Conv_2D(x, filters=512, lrn=True, dilation_rate=1, num_block=4)
    skip_5 = x
    x = Gated_Conv_2D(x, filters=512, lrn=True, dilation_rate=1, num_block=5)
    skip_6 = x
    x = Gated_Conv_2D(x, filters=512, lrn=True, dilation_rate=1, num_block=6)
    skip_7 = x
    x = Gated_Conv_2D(x, filters=512, lrn=True, dilation_rate=1, num_block=7)

    # dilated Gated convolutional layers

    x = Gated_Conv_2D(x, filters=512, lrn=True, dilation_rate=2, strides=1, num_block=8)

    x = Gated_Conv_2D(x, filters=512, lrn=True, dilation_rate=2, strides=1, num_block=9)

    x = Gated_Conv_2D(x, filters=512, lrn=True, dilation_rate=2, strides=1, num_block=10)

    x = Gated_Conv_2D(x, filters=512, lrn=True, dilation_rate=2, strides=1, num_block=11)

    return x, skip_1, skip_2, skip_3, skip_4, skip_5, skip_6, skip_7
def decoder(last_layer, skip_1, skip_2, skip_3, skip_4, skip_5, skip_6, skip_7):
    x_transpose = Conv2DTranspose(filters=512, kernel_size=3, padding="same", strides=2)(last_layer)
    x_transpose = Concatenate(axis=-1)([x_transpose, skip_7])
    x_transpose = Conv2DTranspose(filters=512, kernel_size=3, padding="same", strides=2)(x_transpose)
    x_transpose = Concatenate(axis=-1)([x_transpose, skip_6])
    x_transpose = Conv2DTranspose(filters=512, kernel_size=3, padding="same", strides=2)(x_transpose)
    x_transpose = Concatenate(axis=-1)([x_transpose, skip_5])
    x_transpose = Conv2DTranspose(filters=256, kernel_size=3, padding="same", strides=2)(x_transpose)
    x_transpose = Concatenate(axis=-1)([x_transpose, skip_4])
    x_transpose = Conv2DTranspose(filters=128, kernel_size=3, padding="same", strides=2)(x_transpose)
    x_transpose = Concatenate(axis=-1)([x_transpose, skip_3])
    x_transpose = Conv2DTranspose(filters=64, kernel_size=3, padding="same", strides=2)(x_transpose)
    x_transpose = Concatenate(axis=-1)([x_transpose, skip_2])
    x_transpose = Conv2DTranspose(filters=9, kernel_size=3, padding="same", strides=2)(x_transpose)
    x_transpose = Concatenate(axis=-1)([x_transpose, skip_1])
    x = Gated_Conv_2D(x_transpose, filters=3, lrn=False, dilation_rate=1,strides=1, num_block=12, output_layer=True)

    return x


def generator():
    input = Input(shape=(512, 512, 9))
    last_layer, skip_1, skip_2, skip_3, skip_4, skip_5, skip_6, skip_7 = encoder(input)
    output = decoder(last_layer, skip_1, skip_2, skip_3, skip_4, skip_5, skip_6, skip_7)
    model=Model(inputs=input,outputs=output)
    return model

model=generator()


input = Input(shape=(512, 512, 9))
last_layer, skip_1, skip_2, skip_3, skip_4, skip_5, skip_6, skip_7 = encoder(input)
output=decoder(last_layer, skip_1, skip_2, skip_3, skip_4, skip_5, skip_6, skip_7 )
print("input image",input.shape)
print("output image",output.shape)