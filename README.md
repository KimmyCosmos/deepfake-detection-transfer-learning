# Deepfake Detection with Transfer Learning

## Overview
This project applies transfer learning to a deepfake image classification task by extracting pretrained ResNet18 embeddings and training a PyTorch multilayer perceptron on top of them.

## Methods
- Feature extraction with pretrained ResNet18
- Multilayer perceptron (MLP) classifier
- Stratified train/validation split
- 5-fold cross-validation
- Hyperparameter tuning for learning rate and weight decay
- Test set prediction generation

## Data
Image tensors for binary deepfake classification with separate training and test sets.

## Tools
Python, PyTorch, torchvision, scikit-learn, NumPy, matplotlib
