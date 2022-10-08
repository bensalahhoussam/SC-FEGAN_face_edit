import tensorflow as tf
from keras.layers import Conv2D,LeakyReLU,Input
from keras.models import Model
from sn import SpectralNormalization


inputs = Input(shape=(512,512,3))

class discriminator_block(Model):
    def __init__(self, ):
        super(discriminator_block, self).__init__()
        self.x_1=SpectralNormalization(Conv2D(64*1, (3, 3),strides=1,padding="same", activation=LeakyReLU(alpha=0.1)))
        self.x_2=SpectralNormalization(Conv2D(64*2, (3, 3), strides=2,padding="same",activation=LeakyReLU(alpha=0.1)))
        self.x_3=SpectralNormalization(Conv2D(64*4, (3, 3), strides=2,padding="same",activation=LeakyReLU(alpha=0.1)))
        self.x_4=SpectralNormalization(Conv2D(64*4, (3, 3), strides=2,padding="same",activation=LeakyReLU(alpha=0.1)))
        self.x_5=SpectralNormalization(Conv2D(64*4, (3, 3), strides=2,padding="same",activation=LeakyReLU(alpha=0.1)))
        self.x_6=SpectralNormalization(Conv2D(64*4, (3, 3), strides=2,padding="same",activation=LeakyReLU(alpha=0.1)))

    def call(self, inputs, training=True, **kwargs):
        x=self.x_1(inputs)
        x=self.x_2(x)
        x=self.x_3(x)
        x=self.x_4(x)
        x=self.x_5(x)
        x=self.x_6(x)
        return x


class discriminator_block_gated_conv(Model):
    def __init__(self, ):
        super(discriminator_block, self).__init__()
        self.x_1=SpectralNormalization(Gated_Convolutional(filters=64 * 1, lrn=False, dilation_rate=1, strides=1, num_block=1))
        self.x_2=SpectralNormalization(Gated_Convolutional(filters=64 * 2, lrn=False, dilation_rate=1, strides=2, num_block=2))
        self.x_3=SpectralNormalization(Gated_Convolutional(filters=64 * 4, lrn=False, dilation_rate=1, strides=2, num_block=3))
        self.x_4=SpectralNormalization(Gated_Convolutional(filters=64 * 4, lrn=False, dilation_rate=1, strides=2, num_block=4))
        self.x_5=SpectralNormalization(Gated_Convolutional(filters=64 * 4, lrn=False, dilation_rate=1, strides=2, num_block=5))
        self.x_6=SpectralNormalization(Gated_Convolutional(filters=64 * 4, lrn=False, dilation_rate=1, strides=2, num_block=6))

    def call(self, inputs, training=True, **kwargs):
        x=self.x_1(inputs)
        x=self.x_2(x)
        x=self.x_3(x)
        x=self.x_4(x)
        x=self.x_5(x)
        x=self.x_6(x)
        return x

class Discriminator(Model):
    def __init__(self, gated=False):
        super(Discriminator, self).__init__()
        self.switch=gated
        self.dis_block=discriminator_block()
        self.dis_gated = discriminator_block_gated_conv()
        
    def call(self, inputs, training=True,**kwargs):
        if self.switch == False:
            output=self.dis_block(inputs)
        else: 
            output=self.dis_gated(inputs)
            
        model=Model(inputs=inputs,outputs=output)
        return model
