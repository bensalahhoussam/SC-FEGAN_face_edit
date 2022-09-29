import os
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import tensorflow as tf
import tensorflow_io as tfio
from keras.layers import Input
from build_SNGAN_discriminator import Discriminator
from build_SNGAN_generator import Generator
from SC_FEGAN_data_preparation import Data_Preparation,load_data
from SC_FEGAN_loss_functions import  total_generator_loss,total_dis_loss
from tqdm import tqdm
from tensorflow.keras.optimizers import Adam
path = "D://Deep_Learning_projects/new_projects/computer_vision/project_3/dataset/"
valid_path = "D://CelebAMask-HQ/CelebAMask-HQ/CelebA-HQ-img/validation/"
def model():
    gen_input = Input(shape=(512, 512, 9))
    dis_input = Input(shape=(512, 512, 8))
    model_1 = Generator()
    model_2 = Discriminator()
    gen_model = model_1.call(gen_input)
    dis_model = model_2.call(dis_input)
    return gen_model, dis_model
data = Data_Preparation(path)
def data_split(data):
    input_gen = data.total_input
    ground_truth = data.ground_truth
    batch_data = data.batch_data

    incomplete_image,sketch ,color ,mask,noise = batch_data[0],batch_data[1],batch_data[2],batch_data[3],\
                                                 batch_data[4]

    return input_gen,ground_truth,incomplete_image,sketch,color,mask,noise
def data_distrution(input):
    data=tf.data.Dataset.from_tensor_slices((input)).batch(1,drop_remainder=True)
    return data


model_generator,model_discriminator = model()

generator_optimizer = Adam(1e-4, beta_1=0.1, beta_2=0.999)
discriminator_optimizer = Adam(1e-4, beta_1=0.1, beta_2=0.999)
input_gen,ground_truth,incomplete_image,sketch,color,mask,noise=data_split(data)


input_data = data_distrution(input_gen)
incomplete_data = data_distrution(incomplete_image)
mask_data = data_distrution(mask)
real_data = data_distrution(ground_truth)
sketch_data = data_distrution(sketch)
color_data = data_distrution(color)




def apply_gradient(input_gen,mask,incomplete_image,ground_truth,sketch,color):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

        output_gen = model_generator(input_gen)

        complete_image = incomplete_image + (mask * output_gen)

        batch_pos= tf.concat([ground_truth,sketch,color,mask],axis=-1)
        batch_neg = tf.concat([complete_image,sketch,color,mask], axis=-1)

        dis_real = model_discriminator(batch_pos)
        dis_fake = model_discriminator(batch_neg)

        gen_loss = total_generator_loss(output_gen, ground_truth, complete_image, mask, dis_fake, dis_real)
        dis_loss = total_dis_loss(dis_real,dis_fake,model_discriminator,batch_pos,batch_neg,mask)

        generator_gradients = gen_tape.gradient(gen_loss,model_generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(dis_loss,model_discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(generator_gradients,model_generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(discriminator_gradients,model_discriminator.trainable_variables))


        return output_gen,gen_loss,dis_loss

def train_data_for_one_epoch():
    dis_losses = []
    gen_losses = []

    pbar = tqdm(total=len(list(enumerate(input_data))), position=0, leave=True,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} ')
    for step, data in enumerate(zip(real_data, input_data, incomplete_data, mask_data, sketch_data, color_data)):
        output_gen, gen_loss, dis_loss = apply_gradient(data[1],data[3],data[2],data[0],data[4],data[5])


        gen_losses.append(gen_loss.numpy())
        dis_losses.append(dis_loss.numpy())

        pbar.set_description(f"Training loss for step_num {step} ,gen_loss:{gen_loss:0.4f},dis_loss:{dis_loss:0.4f}")
        pbar.update()


    return gen_losses,dis_losses


def plot_history(epochs_gen_losses, epochs_dis_losses,num_epoch):
    x=list(range(num_epoch))
    plt.figure(figsize=(15,15))
    plt.plot(x, epochs_gen_losses,label='gen_loss',color='green',marker='o')
    plt.plot(x, epochs_dis_losses, label='dis_loss',color='red',marker='x')
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.show()
    plt.savefig('dis_gen_loss.png')
def data_evaluation(i):
    data_load = load_data("D://CelebAMask-HQ/validation/" + str(i) + ".jpg")
    image, incomplete_image, sketch_input, color_input, mask, noise_input = data_load.get_data()
    image = cv.cvtColor(image,cv.COLOR_BGR2RGB)
    incomplete_image  = incomplete_image /127.5 - 1
    incomplete_image = np.expand_dims(incomplete_image,axis=0)

    sketch_input = sketch_input /255.
    sketch_input = np.expand_dims(sketch_input,axis=0)

    color_input = color_input /127.5 - 1.
    color_input = np.expand_dims(color_input,axis=0)

    mask = mask /255.
    mask = np.expand_dims(mask,axis=0)

    noise_input = noise_input /255.
    noise_input = np.expand_dims(noise_input,axis=0)

    batch_input = np.concatenate([incomplete_image, sketch_input, color_input, mask, noise_input],axis=-1)
    output_gen = model_generator(batch_input)

    generated_image = np.squeeze(output_gen,axis=0)+1
    generated_image= cv.cvtColor((generated_image*127.5).astype("uint8"),cv.COLOR_BGR2RGB)

    incomplete_image = np.squeeze(incomplete_image,axis=0)+1
    incomplete_image = cv.cvtColor((incomplete_image *127.5).astype("uint8"),cv.COLOR_BGR2RGB)
    plt.figure(figsize=(10,10))
    plt.subplot(1,3,1)
    plt.imshow(image)
    plt.xlabel("original")
    plt.subplot(1, 3, 2)
    plt.imshow(incomplete_image)
    plt.xlabel("incomplete_image")
    plt.subplot(1, 3, 3)
    plt.imshow(generated_image)
    plt.xlabel("output_image")
    plt.show()


def training_fit(epochs=1):
    epochs_gen_losses, epochs_dis_losses = [], []
    for epoch in range(epochs):
        print(f"Start of epoch number : {epoch+1}")
        start_time = time.time()
        gen_losses,dis_losses = train_data_for_one_epoch()
        elapsed = time.time() - start_time
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        epochs_gen_losses.append(np.mean(gen_losses))
        epochs_dis_losses.append(np.mean(dis_losses))
        print(f'Epoch {epoch+1}: gen_loss: {np.mean(gen_losses):0.3f}  dis_loss: {np.mean(dis_losses):.3f} time:({minutes} min {seconds} sec)')
        os.mkdir(f"model_weights/epoch_{epoch+1}")
        model_generator.save_weights(f"model_weights/epoch_{epoch+1}/gen_model_epoch_{epoch+1}.h5")
        model_discriminator.save_weights(f"model_weights/epoch_{epoch+1}/dis_model_epoch_{epoch+1}.h5")
        genereta_random_numper =np.random.randint(0,15)
        data_evaluation(genereta_random_numper)

    plot_history(epochs_gen_losses,epochs_dis_losses,epochs)


training_fit()






