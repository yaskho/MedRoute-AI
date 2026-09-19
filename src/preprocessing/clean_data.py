import pandas as pd
import numpy as np
import tensorflow as tf

def clean_tabular_features(features_list):
    """
    Prépare ou valide les caractéristiques tabulaires (ex: données du patient).
    """
    # S'assure que les données sont au bon format numérique
    return np.array(features_list, dtype=float).reshape(1, -1)

def preprocess_image_input(image_path, target_size=(224, 224)):
    """
    Charge et prétraite une image pour le modèle MobileNetV2 (mélanome).
    """
    img = tf.keras.utils.load_img(image_path, target_size=target_size)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    return img_array
