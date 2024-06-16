import os
import gdown
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from flask import Flask, request, render_template, redirect, url_for
import numpy as np

app = Flask(__name__)

BATCH_SIZE = 32
IMAGE_SIZE = 299
CHANNELS = 3
NUM_CLASSES = 12
INPUT_SHAPE = (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
class_names = ['Abnormal', 'Erythrodermic', 'Guttate', 'Inverse', 'Nail', 'Normal', 'Not Define', 'Palm Soles', 'Plaque', 'Psoriatic Arthritis', 'Pustular', 'Scalp']

# Download model files from Google Drive
model_architecture_id = '13qxsIj4JLJRFJzl3RwXz4ggQzfiEGql7'
model_weights_id = '1XnTblJ4xx74wcXXtGD0_wO_CgvyKQqb6'
local_model_architecture = 'models/'
local_model_weights = 'models/'

if not os.path.exists(local_model_architecture):
    gdown.download(f'https://drive.google.com/uc?id={model_architecture_id}', local_model_architecture, quiet=False)

if not os.path.exists(local_model_weights):
    gdown.download(f'https://drive.google.com/uc?id={model_weights_id}', local_model_weights, quiet=False)

# Load model
with open(f'{local_model_architecture}Acc97.json', 'r') as json_file:
    model_json = json_file.read()

model = tf.keras.models.model_from_json(model_json)
model.load_weights(f'{local_model_weights}Acc97.weights.h5')

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)
    return predicted_class, confidence

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
    app.run(debug=True)
