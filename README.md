# Liver Disease Prediction System

## Overview
This is a web-based machine learning application designed to predict liver diseases from medical images (such as MRI and CT scans). The system utilizes a ResNet50 deep learning model to classify images into different conditions (e.g., Healthy, Fatty Liver, Cirrhosis, Liver Tumor) and provides visual explanations using Grad-CAM heatmaps.

## Features
- **Deep Learning Classification**: Powered by a fine-tuned ResNet50 model (supports binary or multi-class predictions).
- **Explainable AI (XAI)**: Generates Grad-CAM heatmaps to highlight the regions of the image that contributed most to the model's prediction.
- **DICOM Support**: Natively processes and converts medical `.dcm` files to viewable images.
- **YOLOv8 Integration**: Integration for bounding box object detection (if YOLOv8 weights are available).
- **Responsive Web Interface**: User-friendly frontend built with HTML, CSS, and JavaScript for uploading and analyzing medical images.

## Project Structure
- `app.py`: Main Flask application that serves the API, handles model inference, and serves the frontend.
- `train_model.py`: Script for training and fine-tuning the ResNet50 model using TensorFlow/Keras on a custom dataset.
- `frontend/`: Contains the user interface files (`index.html`, `style.css`, `script.js`).
- `resnet50_liver_model.keras`: The pre-trained Keras model used for classification.
- `requirements.txt`: List of Python dependencies required to run the project.

## Setup and Installation
1. **Navigate to the project directory**.
2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```
3. **Install the dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
### Running the Web Application
1. Start the Flask server:
   ```bash
   python app.py
   ```
2. Open your web browser and navigate to `http://127.0.0.1:5000/`.
3. Upload an image (`.png`, `.jpg`, `.jpeg`, or `.dcm`) to get a prediction and view its corresponding Grad-CAM heatmap.

### Training the Model
To train the model from scratch on your own dataset:
1. Place your training dataset in the `dataset/train/` directory, organized by class labels (e.g., `Fatty Liver/`, `Healthy/`, `[LIVER CIRRHOSIS]/`).
2. Run the training script:
   ```bash
   python train_model.py
   ```
3. The newly trained model will be saved as `resnet50_liver_model.keras` in the root directory.

## Requirements
- Python 3.8+
- TensorFlow 2.15.0
- Flask 3.0.0
- OpenCV
- pydicom
- scikit-learn
- ultralytics (YOLOv8)
- Pillow
