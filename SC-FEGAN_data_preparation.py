import math
import cv2 as cv
import numpy as np

image_path = "D://CelebAMask-HQ/CelebAMask-HQ/CelebA-HQ-img/1255.jpg"


def create_random_mask(input_size=512, number_of_mask=5, max_line=60, max_angle=90, max_length=60):
    mask = np.zeros((input_size, input_size, 1),dtype=np.uint8)
    for i in range(number_of_mask):
        start_x = np.random.randint(input_size - 200)
        start_y = np.random.randint(input_size - 200)
        start_angle = np.random.randint(180)
        numV = np.random.randint(30,max_line)
        for j in range(numV):

            angleP = np.random.randint(-1 * max_angle, max_angle)
            if j % 2 == 0:
                angle = start_angle + angleP
            else:
                angle = start_angle + angleP + 180
            length = np.random.randint(30,max_length)
            end_x = int(start_x + length * math.sin(angle))
            end_y = int(start_y + length * math.cos(angle))
            factor_of_x = int(start_x * 0.1)
            factor_of_y = int(start_y * 0.1)

            cv.line(mask, (start_x + factor_of_x + j, start_y + factor_of_y + j), (end_x, end_y), 255, 15)

    return mask






def edge_detection(image_path):
    img = cv.imread(image_path)
    img = cv.medianBlur(img, 3)
    for _ in range(20):
        img = cv.bilateralFilter(img, 1, 0.1, 6)
    R, G, B = cv.split(img)

    output1_R = cv.equalizeHist(R)
    output1_G = cv.equalizeHist(G)
    output1_B = cv.equalizeHist(B)

    merge = cv.merge((output1_R, output1_G, output1_B))

    (H, W) = merge.shape[:2]
    blob = cv.dnn.blobFromImage(merge, scalefactor=1.0, size=(W, H),
                                swapRB=False, crop=False)
    net = cv.dnn.readNetFromCaffe("deploy.prototxt.txt", "hed_pretrained_bsds.caffemodel")
    net.setInput(blob)
    hed = net.forward()
    hed = cv.resize(hed[0, 0], (512, 512))
    hed = (255 * hed).astype("uint8")
    ret2, th2 = cv.threshold(hed, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    mask = np.reshape(th2, (512, 512, 1))

    return mask


def data_preparation(image_path):
    image = cv.imread(image_path)
    input_image = cv.resize(image, (512, 512))
    noise = np.random.normal(size=(512, 512, 1))
    binary_mask = create_random_mask()
    mask_inv=cv.bitwise_not(binary_mask)
    masked=cv.bitwise_and(input_image,input_image,mask=mask_inv)
    sketch = edge_detection(image_path)
    sketch = sketch * binary_mask
    noise *= binary_mask

    return masked, binary_mask, sketch, noise

masked_image, binary_mask, sketch, noise = data_preparation(image_path)
total_image=np.concatenate([masked_image,masked_image],axis=1)
cv.imshow('img',total_image)
cv.waitKey()

print("input_image", masked_image.shape)

print("noise", noise.shape)

print("mask", binary_mask.shape)

print("sketch", sketch.shape)
