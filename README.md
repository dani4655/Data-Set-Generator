# Data-Set-Generator
Generating data set to train a neural network

# Screen Defects Detection Dataset

## Overview
This repository contains a synthetic dataset for training a neural network to identify screen defects. The dataset includes three types of defects: bubbles, scratches, and dirt stains. Each image in the dataset is labeled with its corresponding defect type.

## Dataset Generation
The dataset is generated using Python and OpenCV. The `data_set_generator.py` script creates synthetic images with defects such as ellipses for bubbles, irregular lines for scratches, and blob-like shapes for dirt stains.

## Usage
- The `data_set_generator.py` script can be modified to adjust the parameters of the synthetic dataset generation.
- The generated dataset can be used to train and test a neural network for screen defect detection.

## Dataset Structure
The dataset is organized into two main folders:
- `images`: Contains the generated images.
- `labels`: Contains the corresponding labels for each image.

## Getting Started
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/screen-defects-dataset.git
   cd screen-defects-dataset
