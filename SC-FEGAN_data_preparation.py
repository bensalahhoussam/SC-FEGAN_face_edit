import math
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from mtcnn.mtcnn import MTCNN
from utils import FaceParser
detector = MTCNN()
image_path = "D://CelebAMask-HQ/CelebAMask-HQ/CelebA-HQ-img/1855.jpg"


def get_hair_mask(pic):
    parser = FaceParser()
    img = pic[..., ::-1]
    parsed = parser.parse_face(img, with_detection=False)
    component_mask = np.zeros(tuple(img.shape[:-1]))
    component_mask[parsed[0] == 17] = 1
    component_mask = np.reshape(component_mask, (img.shape[0], img.shape[1], 1))


    return component_mask.astype("float32")

def create_mask(pic, number_of_mask=5):
    panel_mask = np.zeros((pic.shape[0], pic.shape[1], 1), dtype="float32")
    faces = detector.detect_faces(pic)
    x1, y1, width, height = faces[0]["box"]
    x2, y2 = x1 + width, y1 + height
    mask = np.zeros((y2 - y1, x2 - x1, 1), dtype="int32")

    for i in range(number_of_mask):
        start_x = np.random.randint(0, x2 - x1)
        start_y = np.random.randint(0, y2 - y1)
        start_angle = np.random.randint(360)
        numV = np.random.randint(30, 80)
        for j in range(numV):
            angleP = np.random.randint(-20, 20)
            if j % 2 == 0:
                angle = start_angle + angleP
            else:
                angle = start_angle + angleP + 180
            length = np.random.randint(80, 100)
            end_x = start_x + int(length * math.cos(math.radians(angle)))
            end_y = start_y + int(length * math.sin(math.radians(angle)))

            cv.line(mask, (start_x, start_y), (end_x, end_y), 255, 15)
            start_x = end_x
            start_y = end_y

    panel_mask[y1:y2, x1:x2] = mask[:, :]
    hair_mask_num = np.random.randint(0, 8)
    if hair_mask_num > 5:
        hair_mask = get_hair_mask(pic)
        panel_mask += hair_mask

    panel_mask=np.where(np.clip(panel_mask, 0., 255.) == 0, 0., 255.)/255.
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
    mask = np.reshape(th2, (pic.shape[0], pic.shape[1], 1))
    return mask

def complete_image(output_gen):
    image = image_input+(mask*output_gen)
    return image

class load_data:
    def __init__(self,image_path):
        self.image=image_path

    def get_data(self):

        image = cv.imread(self.image)
        mask = create_mask(image)
        sketch = get_sketch(image)
        color = get_color(image)
        noise = np.random.normal(size=(1024,1024,1))

        reversed_mask=1-mask
        image_input=image*reversed_mask
        sketch_input=sketch*mask
        color_input=color*mask
        noise_input=noise*mask
        return image_input,sketch_input,color_input,mask,noise_input



data=load_data(image_path)
image_input,sketch_input,color_input,mask,noise_input=data.get_data()
