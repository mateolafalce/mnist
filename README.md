# MNIST Digit Classifier with PyTorch

This project implements a simple Convolutional Neural Network (CNN) using PyTorch to classify handwritten digits from the MNIST dataset.

## Overview

- **Dataset:** [MNIST](http://yann.lecun.com/exdb/mnist/) (images of handwritten digits 0-9)
- **Framework:** PyTorch
- **Model:** 2 convolutional layers + 2 fully connected layers
- **Goal:** Classify 28x28 grayscale images into one of 10 digit classes

## How It Works

1. **Data Loading & Preprocessing**
   - Downloads the MNIST dataset.
   - Applies normalization and converts images to PyTorch tensors.
   - Loads data into training and test DataLoaders.

2. **Model Architecture**
   - Two convolutional layers with ReLU activation and max pooling.
   - Flattening followed by a fully connected layer with dropout.
   - Final output layer with 10 units (one per digit class).

3. **Training**
   - Uses CrossEntropyLoss and Adam optimizer.
   - Trains for 5 epochs, printing the average loss per epoch.

4. **Evaluation**
   - Evaluates the trained model on the test set.
   - Prints the overall test accuracy.

## Usage

1. **Requirements**
   - Python 3.x
   - PyTorch
   - torchvision

2. **Run the script**

```bash
python mnist.py
```