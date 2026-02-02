# Defect Detection System

A computer vision-based defect detection system that classifies products as defective or non-defective and highlights the defective regions with bounding boxes.

## Overview

This project uses deep learning to detect defects in product images. It provides:

- Binary classification (defective/non-defective)
- Defect localization with bounding box coordinates
- Explainable AI approach using feature map analysis
- Web interface for easy interaction

## How It Works

The system uses a modified VGG19 architecture:

1. **VGG19 Base**: Pre-trained ImageNet model without top layers extracts visual features
2. **Global Average Pooling**: Reduces feature maps to scalar values
3. **Dense Layers**: Classification head for binary prediction

### Defect Localization

The explainability comes from analyzing the convolutional feature maps:

1. Extract feature maps from the last conv layer (block5_conv3)
2. Multiply each feature map by its corresponding weight from the dense layer
3. Generate a heatmap showing important regions
4. Apply thresholding to identify defect boundaries
5. Draw bounding box around the defective area

## Project Structure

```
defect_detection/
├── app.py                 # Flask web application
├── main.py                # Entry point for testing
├── requirements.txt       # Python dependencies
├── Procfile              # Deployment configuration
├── setup.py              # Package setup
├── defect_detection/     # Core module
│   ├── config.py         # Configuration and model loading
│   └── utils/
│       ├── helpers.py    # Image processing utilities
│       └── prediction.py # Prediction and bbox extraction
├── notebook/             # Training notebooks
│   ├── Bangle_defect_detection.ipynb
│   └── Marble_Defect_detection.ipynb
├── templates/            # HTML templates
│   ├── index.html
│   └── result.html
├── static/               # Static files and output images
└── assets/               # Documentation images
```

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd Defect_detection-main
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dependencies

- Flask - Web framework
- TensorFlow - Deep learning framework
- Keras - Neural network API
- OpenCV - Image processing
- scikit-image - Image transformations
- NumPy - Numerical operations
- Pillow - Image handling
- Matplotlib - Visualization
- gdown - Google Drive model downloads

## Usage

### Running the Web Application

```bash
python app.py
```

Access the application at `http://localhost:5000`

### Using the API

**Endpoints:**

- `GET /` - Home page with upload interface
- `POST /predict` - Submit image for defect detection
- `GET /test` - Health check endpoint

### Training New Models

Use the Jupyter notebooks in the `notebook/` directory:

- `Bangle_defect_detection.ipynb` - Training for bangle products
- `Marble_Defect_detection.ipynb` - Training for marble products

## Configuration

Key settings in `defect_detection/config.py`:

- `IMAGE_SIZE`: Input image dimensions (default: 256x256)
- `THRESHOLD`: Defect detection threshold (default: 0.7)
- `RESIZE_FACTOR`: Output image scaling factor

## Model Downloads

Pre-trained models are automatically downloaded from Google Drive on first run. Models are saved to `defect_detection/models/`.

## Adding New Products

1. Train a new model using the notebook templates
2. Upload the `.h5` model file to Google Drive
3. Add the model ID to `MODEL_DRIVE_IDS` in `config.py`

## API Response

The prediction endpoint returns:
- Classification probabilities (defective vs non-defective)
- Bounding box coordinates (pt1, pt2) for defective regions
- Processed images saved to `static/` folder

## Tech Stack

- **Backend**: Python, Flask
- **ML Framework**: TensorFlow/Keras
- **Base Model**: VGG19 (ImageNet weights)
- **Image Processing**: OpenCV, scikit-image, Pillow
