import os
import gdown
import logging

logging.basicConfig(level=logging.INFO)

def download_from_google_drive(file_id, file_name):
    try:
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, file_name, quiet=False)
        logging.info(f"Downloaded {file_name} from Google Drive")
    except Exception as e:
        logging.error(f"Failed to download {file_name} from Google Drive: {e}")
        raise

if __name__ == "__main__":
    model_architecture_id = os.getenv('13qxsIj4JLJRFJzl3RwXz4ggQzfiEGql7')
    model_weights_id = os.getenv('1XnTblJ4xx74wcXXtGD0_wO_CgvyKQqb6')
    local_model_architecture = '../models/Acc97.json'
    local_model_weights = '../models/Acc97.weights.h5'

    if not os.path.exists(local_model_architecture):
        download_from_google_drive(model_architecture_id, local_model_architecture)

    if not os.path.exists(local_model_weights):
        download_from_google_drive(model_weights_id, local_model_weights)
