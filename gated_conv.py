import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation, Input, Conv2D, LeakyReLU, Multiply, Conv2DTranspose, Concatenate
from tensorflow.keras.utils import plot_model



def Gated_Conv_2D(x_layer, filters, lrn=True, dilation_rate=1, strides=2, num_block=1,act=True):
    x_feature = Conv2D(filters=filters, kernel_size=3, padding="same", strides=strides, dilation_rate=dilation_rate,
                       name="conv_1_" + str(num_block))(x_layer)
    if lrn is True:
        x_feature = tf.nn.local_response_normalization(x_feature,  bias=0.00005,
                                       name="LRN_1_" + str(num_block))
    if act==True:
        x_feature = LeakyReLU(alpha=0.1)(x_feature)


    x_gating = Conv2D(filters=filters, kernel_size=3, padding="same", strides=strides,
                      dilation_rate=dilation_rate, name="gated_conv_1_" + str(num_block))(x_layer)
    x_gating = Activation("sigmoid", name="sigmoid_1_" + str(num_block))(x_gating)


    output = Multiply(name="multiplication_layer_1_" + str(num_block))([x_gating, x_feature])

    return output

def Gated_deconv_2d(x_layer, filters, dilation_rate=1, strides=2, num_block=1, lrn=True):
    x_feature = Conv2DTranspose(filters=filters, kernel_size=3, padding="same", strides=strides,
                                dilation_rate=dilation_rate,use_bias=True,name="deconv_1_" + str(num_block))(x_layer)
    if lrn is True:
        x_feature = tf.nn.local_response_normalization(x_feature,  bias=0.00005,
                                                      name="LRN_1_" + str(num_block))

    x_feature = LeakyReLU(alpha=0.1)(x_feature)

    x_gating = Conv2DTranspose(filters=filters, kernel_size=3, padding="same", strides=2,
                      dilation_rate=dilation_rate, name="gated_deconv_1_" + str(num_block))(x_layer)
    x_gating = Activation("sigmoid", name="de_sigmoid_1_" + str(num_block))(x_gating)


    output = Multiply(name="de_multiplication_layer_1_" + str(num_block))([x_gating, x_feature])
    return output

