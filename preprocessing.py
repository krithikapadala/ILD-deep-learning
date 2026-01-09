import cv2
import numpy as np
import os

def preprocess_image(image_path, img_size=224):
    """
    Reads and preprocesses an image
    """
    image = cv2.imread(image_path)
    image = cv2.resize(image, (img_size, img_size))
    image = image / 255.0
    return image

def load_images_from_folder(folder_path):
    images = []
    for file in os.listdir(folder_path):
        img_path = os.path.join(folder_path, file)
        image = preprocess_image(img_path)
        images.append(image)
    return np.array(images)
