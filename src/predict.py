import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import json
import os
import gdown

BATCH_SIZE = 32
IMAGE_SIZE = 299
CHANNELS = 3
NUM_CLASSES = 12
INPUT_SHAPE = (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
class_names = ['Abnormal', 'Erythrodermic', 'Guttate', 'Inverse', 'Nail', 'Normal', 'Not Define', 'Palm Soles', 'Plaque', 'Psoriatic Arthritis', 'Pustular', 'Scalp']

def download_from_google_drive(file_id, file_name):
    url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(url, file_name, quiet=False)

def predict(model, img_path):
    img = image.load_img(img_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)
    return predicted_class, confidence

if __name__ == "__main__":
    # Google Drive file IDs (replace these with your actual file IDs)
    model_architecture_id = '13qxsIj4JLJRFJzl3RwXz4ggQzfiEGql7'
    model_weights_id = '1XnTblJ4xx74wcXXtGD0_wO_CgvyKQqb6'
    local_model_architecture = '../models/'
    local_model_weights = '../models/'

    # Download model files from Google Drive
    download_from_google_drive(model_architecture_id, local_model_architecture)
    download_from_google_drive(model_weights_id, local_model_weights)

    # Load model
    with open(f'{local_model_architecture}Acc97.json', 'r') as json_file:
        model_json = json_file.read()

    model = tf.keras.models.model_from_json(model_json)
    model.load_weights(f'{local_model_weights}Acc97.weights.h5')

    img_path = "../data/test_image.jpg"
    predicted_class, confidence = predict(model, img_path)
    print(f"Predicted class: {predicted_class} with confidence {confidence}%")
