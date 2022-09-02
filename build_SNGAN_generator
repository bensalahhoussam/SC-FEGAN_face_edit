import tensorflow as tf
from tensorflow.keras.models import Model
from gated_conv import Gated_Conv_2D,Gated_deconv_2d
from tensorflow.keras.layers import Activation, Input, Conv2D, LeakyReLU, Multiply, Conv2DTranspose, Concatenate
from tensorflow.keras.utils import plot_model




def encoder_block(x_layer):
    skip_1 =x_layer
    x = Gated_Conv_2D(x_layer,filters=64*1, lrn=False, dilation_rate=1,strides=2, num_block=1)
    skip_2 = x
    x = Gated_Conv_2D(x,filters=64*2, lrn=True, dilation_rate=1,strides=2, num_block=2)
    skip_3 = x
    x = Gated_Conv_2D(x, filters=64*4, lrn=True, dilation_rate=1,strides=2,num_block=3)
    skip_4 = x
    x = Gated_Conv_2D(x, filters=64*8, lrn=True, dilation_rate=1, num_block=4)
    skip_5 = x
    x = Gated_Conv_2D(x, filters=64*8, lrn=True, dilation_rate=1, num_block=5)
    skip_6 = x
    x = Gated_Conv_2D(x, filters=64*8, lrn=True, dilation_rate=1, num_block=6)
    skip_7 = x
    x = Gated_Conv_2D(x, filters=64*8, lrn=True, dilation_rate=1, num_block=7)
    # dilated Gated convolutional layers
    x = Gated_Conv_2D(x, filters=64*8, lrn=True, dilation_rate=2, strides=1, num_block=8)

    x = Gated_Conv_2D(x, filters=64*8, lrn=True, dilation_rate=2, strides=1, num_block=9)

    x = Gated_Conv_2D(x, filters=64*8, lrn=True, dilation_rate=2, strides=1, num_block=10)

    x = Gated_Conv_2D(x, filters=64*8, lrn=True, dilation_rate=2, strides=1, num_block=11)
    return x, skip_1, skip_2, skip_3, skip_4, skip_5, skip_6, skip_7
def decoder_block(x, skip_1, skip_2, skip_3, skip_4, skip_5, skip_6, skip_7):
    x = Gated_deconv_2d(x, filters=64*8, lrn=True, dilation_rate=1, num_block=1)
    x = Concatenate(axis=-1)([x, skip_7])
    x = Gated_Conv_2D(x, filters=64*8, lrn=True, dilation_rate=1,strides=1, num_block=12)

    x = Gated_deconv_2d(x, filters=64*8, lrn=True, dilation_rate=1, num_block=2)
    x = Concatenate(axis=-1)([x, skip_6])
    x = Gated_Conv_2D(x, filters=64*8, lrn=True, dilation_rate=1,strides=1, num_block=13)

    x = Gated_deconv_2d(x, filters=64*8, lrn=True, dilation_rate=1, num_block=3)
    x = Concatenate(axis=-1)([x, skip_5])
    x = Gated_Conv_2D(x, filters=64*8, lrn=True, dilation_rate=1,strides=1, num_block=14)

    x = Gated_deconv_2d(x, filters=64*4, lrn=True, dilation_rate=1, num_block=4)
    x = Concatenate(axis=-1)([x, skip_4])
    x = Gated_Conv_2D(x, filters=64*4, lrn=True, dilation_rate=1,strides=1, num_block=15)

    x = Gated_deconv_2d(x, filters=64*2, lrn=True, dilation_rate=1, num_block=5)
    x = Concatenate(axis=-1)([x, skip_3])
    x = Gated_Conv_2D(x, filters=64*2, lrn=True, dilation_rate=1,strides=1, num_block=16)

    x = Gated_deconv_2d(x, filters=64*1, lrn=True, dilation_rate=1, num_block=6)
    x = Concatenate(axis=-1)([x, skip_2])
    x = Gated_Conv_2D(x, filters=64*1, lrn=True, dilation_rate=1,strides=1, num_block=17)

    x = Gated_deconv_2d(x, filters=3, lrn=True, dilation_rate=1, num_block=7)
    x = Concatenate(axis=-1)([x, skip_1])
    x = Gated_Conv_2D(x, filters=3, lrn=False, dilation_rate=1,strides=1, num_block=18,act=False)

    x=Activation("tanh")(x)
    return x


def generator():
    input = Input(shape=(512, 512, 9))
    last_layer, skip_1, skip_2, skip_3, skip_4, skip_5, skip_6, skip_7 = encoder_block(input)
    output = decoder_block(last_layer, skip_1, skip_2, skip_3, skip_4, skip_5, skip_6, skip_7)
    model=Model(inputs=input,outputs=output)
    return model

model=generator()
print(model.summary())
plot_model(model, to_file='model.png',show_shapes=True)
