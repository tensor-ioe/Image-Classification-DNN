# Distributed AlexNet for Image Classification

## Table of Contents
- [About the Project](#about-the-project)
- [AlexNet Architecture](#alexnet-architecture)
- [Distributed Training](#distributed-training)
- [Data Augmentation](#data-augmentation)

---

## About the Project
This repository contains the code for training an image-classification model based on the **AlexNet** architecture.  
To accelerate training and handle large datasets, we employ a distributed training setup using *PyTorch*’s built-in distributed capabilities.

---

## AlexNet Architecture
The core of our model is the **AlexNet** architecture, a pioneering convolutional neural network that significantly advanced image recognition.  
For a detailed understanding, see the original paper:

> **ImageNet Classification with Deep Convolutional Neural Networks**  
> <(https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)>

---

## Distributed Training
Our training process is designed to be highly scalable and efficient. We leverage PyTorch’s distributed features to enable communication and synchronization between separate training processes running on GPUs across different machines (connected via a network switch using TCP).

**Key Features**

- **PyTorch Distributed Data Parallel (DDP)** – efficient model replication and gradient synchronization across GPUs.  
- **TCP Communication Backend** – reliable inter-machine communication.  
- **Multi-GPU Support** – maximizes computational throughput.

---

## Data Augmentation
To enhance generalization and prevent overfitting, we apply the following augmentations at load time (via `torchvision.transforms`):

- **Resizing** – scale images (e.g., 256×256), then center-crop to 224×224.  
- **Random Cropping** – introduce variability in object positioning.  
- **Random Horizontal Flipping** – mirror images to increase diversity.  
- **Normalization** – scale pixel values (e.g., mean 0, std 1).  
- **Color Jittering** *(optional)* – random brightness, contrast, saturation, hue.  
- **Random Rotations** *(optional)* – rotate images by a small degree.  

---
