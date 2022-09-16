import math
import cv2 as cv
import os
import numpy as np
import tensorflow as tf
import tensorflow_io as tfio
from keras.layers import Input

from build_SNGAN_discriminator import Discriminator
from build_SNGAN_generator import Generator
from mtcnn.mtcnn import MTCNN
from utils import FaceParser

detector = MTCNN()

image_path = "D://CelebAMask-HQ/CelebAMask-HQ/CelebA-HQ-img/185.jpg"


def get_hair_mask(pic):
    parser = FaceParser()
    img = pic[..., ::-1]
    parsed = parser.parse_face(img, with_detection=False)
    component_mask = np.zeros(tuple(img.shape[:-1]))
    component_mask[parsed[0] == 17] = 1
    component_mask = np.reshape(component_mask, (img.shape[0], img.shape[1], 1))
    return component_mask.astype("float32")


def create_mask(pic, number_of_mask=3):
    panel_mask = np.zeros((pic.shape[0], pic.shape[1], 1), dtype="float32")
    faces = detector.detect_faces(pic)
    x1, y1, width, height = faces[0]["box"]
    x2, y2 = x1 + width, y1 + height
    binary_mask = np.zeros((y2 - y1, x2 - x1, 1), dtype="int32")

    for i in range(number_of_mask):
        start_x = np.random.randint(0, x2 - x1)
        start_y = np.random.randint(0, y2 - y1)
        start_angle = np.random.randint(180)
        numV = np.random.randint(80, 100)
        for j in range(numV):
            angleP = np.random.randint(-15, 15)
            if j % 2 == 0:
                angle = start_angle + angleP
            else:
                angle = start_angle + angleP + 180
            length = np.random.randint(80, 100)
            end_x = start_x + int(length * math.cos(math.radians(angle)))
            end_y = start_y + int(length * math.sin(math.radians(angle)))

            cv.line(binary_mask, (start_x, start_y), (end_x, end_y), 255, 15)
            start_x = end_x
            start_y = end_y

    panel_mask[y1:y2, x1:x2] = binary_mask[:, :]
    hair_mask_num = np.random.randint(2, 8)
    if hair_mask_num > 5:
        hair_mask = get_hair_mask(pic)
        panel_mask += hair_mask
    panel_mask = np.where(np.clip(panel_mask, 0., 255.) == 0, 0., 255.) / 255.
    return panel_mask.astype("uint8")


def get_color(pic):
    annotation_colors = [
        '0, background', '1, skin', '2, left eyebrow', '3, right eyebrow',
        '4, left eye', '5, right eye', '6, glasses', '7, left ear', '8, right ear', '9, earings',
        '10, nose', '11, mouth', '12, upper lip', '13, lower lip',
        '14, neck', '15, neck_l', '16, cloth', '17, hair', '18, hat'
    ]

    parser = FaceParser()
    img = pic[..., ::-1]
    parsed = parser.parse_face(img, with_detection=False)

    # blurring image
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    imgg = cv.medianBlur(img, 5)
    bilateral = imgg
    for i in range(20):
        bilateral = cv.bilateralFilter(bilateral, 20, 30, 30)
    median_image = np.zeros(bilateral.shape)
    for i in range(len(annotation_colors)):
        component_mask = np.zeros(tuple(pic.shape[:-1]))
        component_mask[parsed[0] == i] = 1
        masked = np.multiply(cv.cvtColor(
            bilateral, cv.COLOR_RGB2BGR), np.expand_dims(component_mask, axis=-1))
        median_image += masked
    median_image = median_image.astype(np.uint8)
    median_image = cv.cvtColor(median_image, cv.COLOR_BGR2RGB)
    return median_image


def get_sketch(pic):
    (H, W) = pic.shape[:2]
    blob = cv.dnn.blobFromImage(pic, scalefactor=1.0, size=(W, H),
                                swapRB=False, crop=False)
    net = cv.dnn.readNetFromCaffe("deploy.prototxt.txt", "hed_pretrained_bsds.caffemodel")
    net.setInput(blob)
    hed = net.forward()
    hed = cv.resize(hed[0, 0], (pic.shape[0], pic.shape[1]))
    hed = (255 * hed).astype("uint8")
    ret2, th2 = cv.threshold(hed, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    binary_mask = np.reshape(th2, (pic.shape[0], pic.shape[1], 1))
    return binary_mask.astype(np.uint8)


class load_data:
    def __init__(self, path_image):
        self.image = path_image

    def get_data(self):
        pic = cv.imread(self.image)
        pic = cv.resize(pic, (512, 512))
        binary_mask = create_mask(pic)
        sketch = get_sketch(pic)
        color = get_color(pic)
        noise = np.random.normal(size=(pic.shape[0], pic.shape[1], 1))
        reversed_mask = 1 - binary_mask
        input_image = pic * reversed_mask
        sketch = sketch * binary_mask
        color = color * binary_mask
        noise = noise * binary_mask
        return pic, input_image, sketch, color, binary_mask * 255, noise


class Data_Preparation:
    def __init__(self, folder_path):
        self.path = folder_path
        self.label, self.total_images, self.total_sketch, self.total_color, self.total_mask, self.total_noise = \
            self.data_load()
        self.ground_truth, self.total_batch = self.data_batch()
        self.incomplete_image = self.total_batch[:][..., 0:3]
        self.sketch = self.total_batch[:][..., 3:4]
        self.color = self.total_batch[:][..., 4:7]
        self.mask = self.total_batch[:][..., 7:8]
        self.noise = self.total_batch[:][..., 8:9]
        self.data = [self.incomplete_image, self.sketch, self.color, self.mask, self.noise]

    def data_load(self, ):
        true_img = []
        images = []
        colors = []
        edges = []
        masks = []
        noises = []

        data_folder = [name for name in os.listdir(self.path)]

        images_color = [img for img in os.listdir(self.path + "/" + data_folder[0])]

        truth = [img for img in os.listdir(self.path + "/" + data_folder[1])]

        images_input = [img for img in os.listdir(self.path + "/" + data_folder[2])]

        images_mask = [img for img in os.listdir(self.path + "/" + data_folder[3])]

        images_noise = [img for img in os.listdir(self.path + "/" + data_folder[4])]

        images_sketch = [img for img in os.listdir(self.path + "/" + data_folder[5])]

        for i in range(len(images_color)):
            actual_path, path, sketch_path, color_path, mask_path, noise_path = truth[i], \
                                                                                images_input[i], \
                                                                                images_sketch[i], \
                                                                                images_color[i], \
                                                                                images_mask[i], \
                                                                                images_noise[i]
            true_img.append(self.path + data_folder[1] + "/" + actual_path)
            images.append(self.path + data_folder[2] + "/" + path)
            edges.append(self.path + data_folder[5] + "/" + sketch_path)
            colors.append(self.path + data_folder[0] + "/" + color_path)
            masks.append(self.path + data_folder[3] + "/" + mask_path)
            noises.append(self.path + data_folder[4] + "/" + noise_path)
        return true_img, images, edges, colors, masks, noises

    def data_batch(self, ):
        total_label = []
        total_batch = []
        for i in range(len(self.total_sketch)):
            actual = tf.image.decode_jpeg(tf.io.read_file(self.label[i]), channels=3)
            actual = tfio.experimental.color.bgr_to_rgb(actual)
            actual = tf.cast(actual, dtype=tf.float32) / 127.5 - 1.

            pic = tf.image.decode_jpeg(tf.io.read_file(self.total_images[i]), channels=3)
            pic = tfio.experimental.color.bgr_to_rgb(pic)
            pic = tf.cast(pic, dtype=tf.float32) / 127.5 - 1.

            sketch = tf.image.decode_jpeg(tf.io.read_file(self.total_sketch[i]), channels=1)
            sketch = tf.cast(sketch, dtype=tf.float32) / 255.

            color = tf.image.decode_jpeg(tf.io.read_file(self.total_color[i]), channels=3)
            color = tfio.experimental.color.bgr_to_rgb(color)
            color = tf.cast(color, dtype=tf.float32) / 255.

            mask = tf.image.decode_jpeg(tf.io.read_file(self.total_mask[i]), channels=1)
            mask = tf.cast(mask, dtype=tf.float32) / 255.

            noise = tf.image.decode_jpeg(tf.io.read_file(self.total_noise[i]), channels=1)
            noise = tf.cast(noise, dtype=tf.float32) / 255.

            batch_input = tf.concat([pic, sketch, color, mask, noise], axis=-1)
            total_batch.append(batch_input)
            total_label.append(actual)

        total_batch = tf.stack(total_batch, axis=0)
        total_label = tf.stack(total_label, axis=0)
        return total_label, total_batch

    def complete_image(self, gen_output):
        pic = self.incomplete_image + (self.mask * gen_output)
        return pic


def model():
    gen_input = Input(shape=(512, 512, 9))
    dis_input = Input(shape=(512, 512, 5))
    model_1 = Generator()
    model_2 = Discriminator()
    gen_model = model_1.call(gen_input)
    dis_model = model_2.call(dis_input)
    return gen_model, dis_model


"""file = "D://Deep_Learning_projects/new_projects/computer_vision/project_3/dataset/"
for i in range(50):
    try:
        data = load_data("D://CelebAMask-HQ/CelebAMask-HQ/CelebA-HQ-img/" + str(i) + ".jpg")
        image, image_input, sketch_input, color_input, mask, noise_input = data.get_data()
        cv.imwrite(file + f"/ground_truth/image_{i}.jpg", image)
        cv.imwrite(file + f"/image_input/image_{i}.jpg", image_input)
        cv.imwrite(file + f"/sketch_input/sketch_{i}.jpg", sketch_input)
        cv.imwrite(file + f"/color_input/color_{i}.jpg", color_input)
        cv.imwrite(file + f"/mask_input/mask_{i}.jpg", mask)
        cv.imwrite(file + f"/noise_input/noise_{i}.jpg", noise_input * 255)
    except:
        pass"""
