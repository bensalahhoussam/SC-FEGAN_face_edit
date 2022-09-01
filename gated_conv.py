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
