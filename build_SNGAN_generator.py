import tensorflow as tf
from tensorflow.keras.models import Model
from gated_conv import Gated_Convolutional,Gated_Deconvolutional
from tensorflow.keras.layers import Activation, Input, Conv2D, LeakyReLU, Multiply, Conv2DTranspose, Concatenate
from tensorflow.keras.utils import plot_model




class Encoder_Block(Model):
    def __init__(self, ):
        super(Encoder_Block, self).__init__()
        self.x_gated_conv_1 = Gated_Convolutional(filters=64 * 1, lrn=False, dilation_rate=1, strides=2, num_block=1)
        self.x_gated_conv_2 = Gated_Convolutional(filters=64 * 2, lrn=True, dilation_rate=1, strides=2, num_block=2)
        self.x_gated_conv_3 = Gated_Convolutional(filters=64 * 4, lrn=True, dilation_rate=1, strides=2, num_block=3)
        self.x_gated_conv_4 = Gated_Convolutional(filters=64 * 8, lrn=True, dilation_rate=1, strides=2, num_block=4)
        self.x_gated_conv_5 = Gated_Convolutional(filters=64 * 8, lrn=True, dilation_rate=1, strides=2, num_block=5)
        self.x_gated_conv_6 = Gated_Convolutional(filters=64 * 8, lrn=True, dilation_rate=1, strides=2, num_block=6)
        self.x_gated_conv_7 = Gated_Convolutional(filters=64 * 8, lrn=True, dilation_rate=1, strides=2, num_block=7)

        # dilated Gated convolutional layers

        self.x_gated_conv_8 = Gated_Convolutional(filters=64 * 8, lrn=True, dilation_rate=2, strides=1, num_block=8)
        self.x_gated_conv_9 = Gated_Convolutional(filters=64 * 8, lrn=True, dilation_rate=2, strides=1, num_block=9)
        self.x_gated_conv_10 = Gated_Convolutional(filters=64 * 8, lrn=True, dilation_rate=2, strides=1, num_block=10)
        self.x_gated_conv_11 = Gated_Convolutional(filters=64 * 8, lrn=True, dilation_rate=2, strides=1, num_block=11)

    def call(self, inputs, training=True, **kwargs):
        skip_1 = inputs
        x = self.x_gated_conv_1(inputs)
        skip_2 = x
        x = self.x_gated_conv_2(x)
        skip_3 = x
        x = self.x_gated_conv_3(x)
        skip_4 = x
        x = self.x_gated_conv_4(x)
        skip_5 = x
        x = self.x_gated_conv_5(x)
        skip_6 = x
        x = self.x_gated_conv_6(x)
        skip_7 = x
        x = self.x_gated_conv_7(x)

        x = self.x_gated_conv_8(x)
        x = self.x_gated_conv_9(x)
        x = self.x_gated_conv_10(x)
        x = self.x_gated_conv_11(x)
        return x, skip_1, skip_2, skip_3, skip_4, skip_5, skip_6, skip_7




class Decoder_Block(Model):
    def __init__(self, ):
        super(Decoder_Block, self).__init__()
        self.x_gated_deconv_1 = Gated_Deconvolutional(filters=64 * 8, lrn=True, dilation_rate=1, strides=2, num_block=1)
        self.concat = Concatenate(axis=-1)
        self.x_gated_conv_12 = Gated_Convolutional(filters=64 * 8, lrn=True, dilation_rate=1, strides=1, num_block=12)

        self.x_gated_deconv_2 = Gated_Deconvolutional(filters=64 * 8, lrn=True, dilation_rate=1, strides=2, num_block=2)
        self.x_gated_conv_13 = Gated_Convolutional(filters=64 * 8, lrn=True, dilation_rate=1, strides=1, num_block=13)

        self.x_gated_deconv_3 = Gated_Deconvolutional(filters=64 * 8, lrn=True, dilation_rate=1, strides=2, num_block=3)
        self.x_gated_conv_14 = Gated_Convolutional(filters=64 * 8, lrn=True, dilation_rate=1, strides=1, num_block=14)

        self.x_gated_deconv_4 = Gated_Deconvolutional(filters=64 * 4, lrn=True, dilation_rate=1, strides=2, num_block=4)
        self.x_gated_conv_15 = Gated_Convolutional(filters=64 * 4, lrn=True, dilation_rate=1, strides=1, num_block=15)

        self.x_gated_deconv_5 = Gated_Deconvolutional(filters=64 * 2, lrn=True, dilation_rate=1, strides=2, num_block=5)
        self.x_gated_conv_16 = Gated_Convolutional(filters=64 * 2, lrn=True, dilation_rate=1, strides=1, num_block=16)

        self.x_gated_deconv_6 = Gated_Deconvolutional(filters=64 * 1, lrn=True, dilation_rate=1, strides=2, num_block=6)
        self.x_gated_conv_17 = Gated_Convolutional(filters=64 * 1, lrn=True, dilation_rate=1, strides=1, num_block=17)

        self.x_gated_deconv_7 = Gated_Deconvolutional(filters=3, lrn=True, dilation_rate=1, strides=2, num_block=7)
        self.x_gated_conv_18 = Gated_Convolutional(filters=3, lrn=True, dilation_rate=1, strides=1, num_block=18,
                                                 activation=False)

        self.tanh = Activation("tanh")

    def call(self, inputs, training=True,**kwargs):
        x = self.x_gated_deconv_1(inputs[0])
        x = self.concat([x,inputs[7]])
        x = self.x_gated_conv_12(x)

        x = self.x_gated_deconv_2(x)
        x = self.concat([x, inputs[6]])
        x = self.x_gated_conv_13(x)

        x = self.x_gated_deconv_3(x)
        x = self.concat([x, inputs[5]])
        x = self.x_gated_conv_14(x)

        x = self.x_gated_deconv_4(x)
        x = self.concat([x, inputs[4]])
        x = self.x_gated_conv_15(x)

        x = self.x_gated_deconv_5(x)
        x = self.concat([x, inputs[3]])
        x = self.x_gated_conv_16(x)

        x = self.x_gated_deconv_6(x)
        x = self.concat([x, inputs[2]])
        x = self.x_gated_conv_17(x)

        x = self.x_gated_deconv_7(x)
        x = self.concat([x, inputs[1]])
        x = self.x_gated_conv_18(x)
        output=self.tanh(x)
        return output

class Generator():
    def __init__(self, ):
        super(Generator, self).__init__()
        self.encoder=Encoder_Block()
        self.decoder=Decoder_Block()
    def call(self, inputs, ,training=True,**kwargs):
        x = self.encoder(inputs)
        output = self.decoder(x)
        model = Model(inputs=inputs, outputs=output)
        return model

model=generator()
print(model.summary())
plot_model(model, to_file='model.png',show_shapes=True)
