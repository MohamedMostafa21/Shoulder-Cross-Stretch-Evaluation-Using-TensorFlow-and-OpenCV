import cv2
import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.preprocessing import image


model = tf.keras.models.load_model('models/shoulder_stretch_model_optimized.keras')

def preprocess_and_predict(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    return prediction

# Test on a few images
print(preprocess_and_predict('data/test/correct/correct_41.jpg'))
print(preprocess_and_predict('data/test/correct/correct2_43.jpg'))
print(preprocess_and_predict('data/test/incorrect/incorrect_42.jpg'))
print(preprocess_and_predict('data/train/incorrect/incorrect_6.jpg'))
print(preprocess_and_predict('data/train/incorrect/incorrect_10.jpg'))
print(preprocess_and_predict('data/train/correct/correct_24.jpg'))
print(preprocess_and_predict('data/test/correct/correct2_43.jpg'))
print(preprocess_and_predict('data/val/correct/correct_35.jpg'))
print(preprocess_and_predict('data/val/incorrect/incorrect_33.jpg'))



