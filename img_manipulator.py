import numpy as np
import cv2

# def normalize_img(image_data):
#     image_min = np.amin(image_data)
#     image_max = np.amax(image_data)
#     a = 0.1
#     b = 0.9
#     return a + (((image_data - image_min) * (b - a)) / (image_max - image_min))


def flip_img(image_data):
    # horizontal_flip
    return cv2.flip(image_data,1)

def bright_image(image):
    bright_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    random_bright = .25 + np.random.uniform()
    bright_image[:, :, 2] = bright_image[:, :, 2] * random_bright
    image1 = cv2.cvtColor(bright_image, cv2.COLOR_HSV2RGB)
    return image1



