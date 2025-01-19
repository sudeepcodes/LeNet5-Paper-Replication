# LeNet5-Paper-Replication

This repository contains the implementation of the LeNet-5 convolutional neural network (CNN) model as described in the original research paper [LeNet-5, Convolutional Neural Networks](http://yann.lecun.com/exdb/lenet/). The model is implemented using the PyTorch framework and is trained and evaluated on the MNIST dataset.

LeNet-5 was one of the earliest and most influential convolutional neural networks, introduced by Yann LeCun et al. in 1998. It was designed for handwritten digit classification and achieved state-of-the-art performance at the time.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Getting Started](#getting-started)
   - [Prerequisites](#prerequisites)
   - [Installation](#installation)
3. [Usage](#usage)
   - [Training the Model](#training-the-model)
   - [Evaluating the Model](#evaluating-the-model)
4. [Model Architecture](#model-architecture)
5. [Results](#results)
6. [Acknowledgments](#acknowledgments)
7. [License](#license)

---

## Project Overview

This project implements the LeNet-5 architecture using PyTorch, replicating the research paper's design for image classification. The LeNet-5 model consists of the following layers:

1. **Input Layer**: 32x32 grayscale images (MNIST images are 28x28, but padding is applied to make them 32x32).
2. **Convolutional Layer 1 (C1)**: 6 filters of size 5x5, followed by average pooling.
3. **Convolutional Layer 2 (C3)**: 16 filters of size 5x5, followed by average pooling.
4. **Fully Connected Layers**:
   - Fully connected layer with 120 units.
   - Fully connected layer with 84 units.
   - Output layer with 10 units (for the 10 classes in MNIST).

The model is trained and tested on the **MNIST dataset**, which consists of 28x28 grayscale images of handwritten digits (0-9).

---

## Getting Started

### Prerequisites

To run this project, you need to have Python and PyTorch installed on your machine. You will also need the `torchvision` library for dataset loading and transformations.

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/LeNet5-Paper-Replication.git
   cd LeNet5-Paper-Replication
   ```   

2. Install requirements

   ```bash
   pip install -r requirements.txt
   ```
   
### Train the model
   Tweak necessary hyperparameters in `config.py`
   ```bash
   python train.py
   ```

### Run Inference
   
   ```bash
   python inference.py
   ```

