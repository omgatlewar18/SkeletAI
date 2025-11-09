"""
Central configuration file for the SkeletAI project.
Contains paths, model hyperparameters, and data settings.
"""
import os

# --- Project Paths ---
# Base directory is the folder where this config.py file is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models') # Points to your 'models' folder
LOG_DIR = os.path.join(BASE_DIR, 'logs')   # Points to your 'logs' folder

# --- Data Parameters ---
IMAGE_SIZE = (224, 224) # Target image size (W, H) as per synopsis
BATCH_SIZE = 32
CLASS_NAMES = ['Female', 'Male'] # Must match folder names in data/train

# --- Model Parameters ---
# Models to compare, as per synopsis (Phase 3)
MODELS_TO_COMPARE = ['VGG16', 'ResNet50', 'MobileNetV2']
# Default model to use for training or evaluation
DEFAULT_MODEL = 'ResNet50' 

# --- Training Parameters ---
EPOCHS = 50
LEARNING_RATE = 1e-4
# Callback parameters
EARLY_STOPPING_PATIENCE = 5
REDUCE_LR_PATIENCE = 3
MIN_LR = 1e-7

# --- Augmentation Parameters ---
# This config can be used by data_preprocessing.py
AUGMENTATION_CONFIG = {
    'rescale': 1./255,
    'rotation_range': 10,
    'width_shift_range': 0.1,
    'height_shift_range': 0.1,
    'shear_range': 0.1,
    'zoom_range': 0.1,
    'horizontal_flip': True,
    'fill_mode': 'nearest'
}

# --- Validation/Test Parameters ---
# Only apply rescaling to validation and test data
VALIDATION_CONFIG = {
    'rescale': 1./255
}

# --- Grad-CAM Parameters ---
# Last convolutional layer for each model
GRADCAM_LAYER_NAMES = {
    'VGG16': 'block5_conv3',
    'ResNet50': 'conv5_block3_out',
    'MobileNetV2': 'out_relu' # MobileNetV2's last conv block output
}

if __name__ == "__main__":
    print("Config file loaded. This file is intended to be imported, not run directly.")

