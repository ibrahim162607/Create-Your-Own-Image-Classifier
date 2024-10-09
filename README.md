# Flower Species Image Classifier
## Overview
This project demonstrates an Image Classifier that can recognize different species of flowers. Using PyTorch and a pre-trained deep learning model, the classifier is trained on the 102 Category Flower Dataset and can predict the species of flowers from images. The final implementation includes a command-line application for training the model on any labeled dataset and predicting flower species from new images.

The model leverages transfer learning by using a pre-trained model (DenseNet121) and fine-tuning it for flower classification. The project also includes utilities for image processing, data loading, and model checkpointing.

## Features
- Command-line Interface (CLI): Allows users to train a new model and make predictions via CLI commands.
- Pre-trained Model: Utilizes transfer learning with DenseNet121.
- Model Checkpointing: Save and load model checkpoints to continue training or perform inference later.
- Top-K Predictions: Display the top K most probable classes for a given image.
- GPU Support: Training and inference can run on GPU if available.

## Table of Contents
- Project Structure
- Setup
- Usage
- Training the Model
- Predicting a Flower Species
- Model Architecture
- Contributing
- License

## Project Structure
.
├── train.py                  # Script to train the model

├── predict.py                # Script to make predictions on a new image

├── utils.py                  # Helper functions for model saving, loading, and data preprocessing

├── cat_to_name.json          # JSON file mapping category numbers to flower names

├── checkpoint.pth            # Example checkpoint file (model weights)

├── README.md                 # This README file

└── LICENSE                   # License information

## Setup

## Requirements
Python 3.x
PyTorch
Pillow (for image processing)
NumPy
Matplotlib (for optional visualizations)
argparse (for command-line argument parsing)

## Model Architecture
The project uses the DenseNet121 architecture, which is a pre-trained model from PyTorch's torchvision.models library. The classifier is fine-tuned for the specific task of classifying flower species. The model checkpoint saves the following:

- Input size: 1024 (features from DenseNet121)
- Output size: 102 (number of flower classes)
- Hidden units: Configurable via command line (train.py)
- Classifier: Fully connected layers followed by ReLU activations and dropout for regularization.

## Transfer Learning
The model uses transfer learning to adapt a pre-trained DenseNet121 to flower species classification. The final layer (classifier) is replaced with a custom classifier trained on the flower dataset.
