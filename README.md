# State Farm Distracted Driver Detection

![Distracted Driver Detection](distracted_driver_detection.jpg)

## Overview

This project aims to develop a deep learning model for detecting distracted drivers using the State Farm Distracted Driver Detection dataset. Leveraging the power of transfer learning with the ResNet50 architecture, we achieve high performance with limited labeled data.

## Getting Started

### Prerequisites

- Python 3
- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib
- Seaborn

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/your-username/state-farm-distracted-driver-detection.git
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. **Data Preparation:**
   - Download the State Farm Distracted Driver Detection dataset.
   - Preprocess the images, including resizing and normalization.

2. **Model Training:**
   - Run the `train_model.py` script to train the distracted driver detection model.
   - Fine-tune the pre-trained ResNet50 model on the dataset.
   - Monitor training progress and evaluate model performance.

3. **Inference:**
   - Use the trained model for inference on new images of distracted drivers.
   - Analyze model predictions and visualize results.

## File Descriptions

- `train_model.py`: Python script for training the distracted driver detection model.
- `evaluate_model.py`: Python script for evaluating the trained model's performance.
- `inference.py`: Python script for performing inference with the trained model.
- `requirements.txt`: File containing the required Python packages.


## Acknowledgments

- State Farm for providing the Distracted Driver Detection dataset.
- TensorFlow and Keras communities for developing deep learning frameworks.
- Contributors to open-source libraries used in this project.
