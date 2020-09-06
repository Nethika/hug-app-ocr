import pytesseract
from PIL import Image
import tensorflow as tf

from keras.models import load_model
from tensorflow import Graph

import os
import json
import cv2
import numpy as np

global class_graph




def classify(img, c_model):
    #global class_graph
    """ classifies images in a given folder using the 'model'"""

    #img = load_img(im_path,target_size=(input_height, input_width))
    #img = img_to_array(img)
    im_size = 128
    # resize 

    img = cv2.resize(img, (im_size,im_size))

    img = img.astype("float") / 255.0
    img = np.expand_dims(img, axis=0)
    with class_graph.as_default():
        predictions = c_model.predict(img)[0]

    return predictions

if __name__ == '__main__':
    im_name = "data/demo/images(1).jpg"
    # load model
    model_path = "data/credit-card.model"
    class_model = load_model(model_path)

    class_graph=tf.get_default_graph()


    crop_img = cv2.imread(im_name)

    predictions = classify(crop_img, class_model)
    print(predictions)