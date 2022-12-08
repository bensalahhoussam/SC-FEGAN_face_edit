## SC-FEGAN_Face Editing Generative Adversarial Network with User's Sketch and Color
![GUI](https://user-images.githubusercontent.com/112108580/194565225-8a8ed270-0baa-468d-8d72-d3bc48404f8a.gif)

## Overview
Edit face images using a a deep neural network. Users can edit face images using intuitive inputs such as sketching and coloring, from which our network SC-FEGAN generates high quality synthetic images. We used SN-patchGAN discriminator and Unet-like generator with gated convolutional layers.

![image](https://user-images.githubusercontent.com/112108580/206371878-32cca246-69ad-4a1e-a00d-979975a69821.png)


## Network Architecture 

The network is based on encoder-decoder architecture like the U-net with gated convolutional layers layers to generate images and discrimination network 
is based on SN-patchGAN.

The network trains generator and discriminator simultaneously. The generator receives incomplete images with user input to create an output imageinthe RGB channel,and inserts the masked area of the output image into the incomplete input image to create a complete image. The discriminator receives either a completed image or an original image (without masking) to determine whether the given input is real or fake.



## Dependencies
* Python 3
* Tensorflow 2.x
* Numpy
* Matplotlib
* TQDM
* OpenCV

## How to Use
* Prepare dataset path in 'path_file.py' to generate training images file list.
* Set the training path and validation path in 'path_file.py'
* Run 'python SNGAN_train.py'
* To run the model in multi GPU run 'SN_GAN_train_GPU.py'

## References 
* https://doi.org/10.48550/arXiv.1804.07723 : Image Inpainting for Irregular Holes Using Partial Convolutions
* https://doi.org/10.48550/arXiv.1806.03589 : Free-Form Image Inpainting with Gated Convolution                                                  
* https://doi.org/10.48550/arXiv.1902.06838 : SC-FEGAN: Face Editing Generative Adversarial Network with User’s Sketch and Color

