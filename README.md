# Disease Classification

This project classifies 12 classes of diseases using a deep learning model built with TensorFlow and Keras. The model achieves 97% accuracy on the test set.

## Directory Structure

- `data/`: Contains placeholders for dataset directories.
- `notebooks/`: Jupyter notebooks for EDA.
- `src/`: Source code for the project.
- `models/`: Directory to save trained models.
- `scripts/`: Shell scripts for running training, evaluation, and prediction.

## Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/ABo-EsMaiL/Psoriasis-Classifications.git
   cd Psoriasis-Classifications
   ```

2. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

3. Place your dataset in the `data` directory. The exact dataset structure is private and should not be shared.

4. Download the model files from Google Drive:
   ```bash
   python src/download_models.py
   ```

## Usage
```
web: python src/app.py
```

### Training

```bash
bash scripts/run_training.sh
```
