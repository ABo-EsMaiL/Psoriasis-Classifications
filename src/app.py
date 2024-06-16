import os
import gdown
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from flask import Flask, request, render_template, redirect, url_for
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

# Set TensorFlow environment variables
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

BATCH_SIZE = 32
IMAGE_SIZE = 299
CHANNELS = 3
NUM_CLASSES = 12
INPUT_SHAPE = (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
class_names = ['Abnormal', 'Erythrodermic', 'Guttate', 'Inverse', 'Nail', 'Normal', 'Not Define', 'Palm Soles', 'Plaque', 'Psoriatic Arthritis', 'Pustular', 'Scalp']

# Google Drive file IDs (replace these with your actual file IDs)
model_architecture_id = '13qxsIj4JLJRFJzl3RwXz4ggQzfiEGql7'
model_weights_id = '1XnTblJ4xx74wcXXtGD0_wO_CgvyKQqb6'
local_model_architecture = 'models/Acc97.json'
local_model_weights = 'models/Acc97.weights.h5'

def download_from_google_drive(file_id, file_name):
    try:
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, file_name, quiet=False)
        logging.info(f"Downloaded {file_name} from Google Drive")
    except Exception as e:
        logging.error(f"Failed to download {file_name} from Google Drive: {e}")
        raise

# Download model files if they don't exist
if not os.path.exists(local_model_architecture):
    download_from_google_drive(model_architecture_id, local_model_architecture)

if not os.path.exists(local_model_weights):
    download_from_google_drive(model_weights_id, local_model_weights)

# Load model
try:
    with open(local_model_architecture, 'r') as json_file:
        model_json = json_file.read()
    model = tf.keras.models.model_from_json(model_json)
    model.load_weights(local_model_weights)
    logging.info("Model loaded successfully")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    raise

def predict_image(img_path):
    try:
        img = image.load_img(img_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        predictions = model.predict(img_array)
        predicted_class = class_names[np.argmax(predictions[0])]
        confidence = round(100 * (np.max(predictions[0])), 2)
        return predicted_class, confidence
    except Exception as e:
        logging.error(f"Error predicting image: {e}")
        raise

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            file_path = os.path.join('uploads', file.filename)
            file.save(file_path)
            predicted_class, confidence = predict_image(file_path)
            return render_template('result.html', predicted_class=predicted_class, confidence=confidence)
    return render_template('index.html')

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
