import gdown

def download_from_google_drive(file_id, file_name):
    url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(url, file_name, quiet=False)

if __name__ == "__main__":
    # Google Drive file IDs
    model_architecture_id = '13qxsIj4JLJRFJzl3RwXz4ggQzfiEGql7'
    model_weights_id = '1XnTblJ4xx74wcXXtGD0_wO_CgvyKQqb6'
    local_model_architecture = '../models/Acc97.json'
    local_model_weights = '../models/Acc97.weights.h5'

    # Download model files from Google Drive
    download_from_google_drive(model_architecture_id, local_model_architecture)
    download_from_google_drive(model_weights_id, local_model_weights)
