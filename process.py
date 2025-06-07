from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import tensorflow as tf

def preprocess_image(image_path, target_size=(225, 225)):
            img = load_img(image_path, target_size=target_size)
            x = img_to_array(img)
            x = x.astype('float32') / 255.
            x = np.expand_dims(x, axis=0)
            return x

def imageAns(image_path):
    model= tf.keras.models.load_model('model.keras')
    img= preprocess_image(image_path)
    ans= model.predict(img)
    return ans 

def imageAnsId(image_path):
    model= tf.keras.models.load_model('plant_ident.h5')
    img= preprocess_image(image_path)
    ans= model.predict(img)
    # print(model.classes_)
    return ans 