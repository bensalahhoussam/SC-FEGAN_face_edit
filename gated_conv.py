import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation, Input, Conv2D, LeakyReLU, Multiply, Conv2DTranspose, Concatenate
from tensorflow.keras.utils import plot_model



class Gated_Convolutional(Model):

    def __init__(self, filters, num_block, strides, dilation_rate, lrn=True, activation=True):
        super(Gated_Convolutional, self).__init__()
        self.lrn = lrn
        self.act = activation
        self.num_block = num_block

        self.x_feature = Conv2D(filters=filters, kernel_size=3, padding="same", strides=strides,
                                dilation_rate=dilation_rate, name="conv_" + str(self.num_block))
        self.activation = LeakyReLU(alpha=0.1)

        self.x_gating = Conv2D(filters=filters, kernel_size=3, padding="same", strides=strides,
                               dilation_rate=dilation_rate, name="gated_conv_" + str(self.num_block))
        self.sigmoid = Activation("sigmoid", name="sigmoid_" + str(self.num_block))

        self.multiply = Multiply(name="multiplication_layer_" + str(self.num_block))

    def call(self, input_tensor, training=True, **kwargs):

        x_feature = self.x_feature(input_tensor)

        if self.lrn:
            x_feature = tf.nn.local_response_normalization(x_feature, bias=0.00005,
                                                           name="LRN_" + str(self.num_block))
        if self.act:
            x_feature = self.activation(x_feature)

        x_gating = self.x_gating(input_tensor)
        x_gating = self.sigmoid(x_gating)

        output = self.multiply([x_gating, x_feature])

        return output


class Gated_Deconvolutional(Model):
    def __init__(self, filters, num_block, strides, dilation_rate, lrn=True):
        super(Gated_Deconvolutional, self).__init__()
        self.lrn = lrn
        self.num_block = num_block

        self.x_feature = Conv2DTranspose(filters=filters, kernel_size=3, padding="same", strides=strides,
                                         dilation_rate=dilation_rate, use_bias=True, name="deconv_" + str(num_block))

        self.activation = LeakyReLU(alpha=0.1)

        self.x_gating = Conv2DTranspose(filters=filters, kernel_size=3, padding="same", strides=2,
                                        dilation_rate=dilation_rate, name="gated_deconv_" + str(num_block))

        self.sigmoid = Activation("sigmoid", name="sigmoid_" + str(self.num_block))

        self.multiply = Multiply(name="multiplication_layer_" + str(self.num_block))

    def call(self, input_tensor, training=True, **kwargs):
        x_feature = self.x_feature(input_tensor)
        if self.lrn:
            x_feature = tf.nn.local_response_normalization(x_feature, bias=0.00005,
                                                           name="LRN_" + str(self.num_block))
        x_feature = self.activation(x_feature)
        x_gating=self.x_gating(input_tensor)
        x_gating=self.sigmoid(x_gating)
        output=self.multiply([x_gating,x_feature])
        return output
        
