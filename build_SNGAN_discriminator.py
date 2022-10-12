import tensorflow as tf
from keras.layers import Conv2D,LeakyReLU,Input
from keras.models import Model
from spectral_normalization import SpectralNormalization
from gated_conv import Gated_Convolutional


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



   

class Discriminator(Model):
    def __init__(self, gated=False):
        super(Discriminator, self).__init__()
        self.switch=gated
        self.dis_block=discriminator_block()
        
        
    def call(self, inputs, training=True,**kwargs):
        output=self.dis_block(inputs)
        model=Model(inputs=inputs,outputs=output)
        return model
