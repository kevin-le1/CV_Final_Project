# Water Pollutant Detection and Classification

This repository contains the code and resources for the research project "Machine Learning for Water Pollutant Detection and Classification." The project aims to develop a robust computer vision system leveraging machine learning techniques to accurately detect and classify various types of water pollutants from images and videos captured in rivers, lakes, and coastal waters.

## Project Overview

The system employs a two-stage approach:

1. **Object Detection**: Localize pollutants within images or video frames using state-of-the-art object detection models such as YOLOv8 architecture and adam for momentum.
2. **Multi-class Classification**: Categorize the images into predefined classes (e.g., clean water, spils (oil), unclean water) using deep neural network architectures like ResNet and EfficientNet.

The project explores strategies to handle challenges such as varying lighting conditions, water turbidity, occlusions, and class imbalances, employing techniques like data augmentation, transfer learning, ensemble methods, and attention mechanisms.

## Repository Structure

- `cnn/`: Directory for our CNN model and results

- `obj_det_no_gpu/`: Directory for our object detection model and results (trained w/o gpu data stored in modal labs)
  - `obj_det_no_gpu/results/`: First trained results parameters 1280 720 batches 4 epochs 10 lr 1e-4 (has evaluation / model)
  - `obj_det_no_gpu/results2/`: Second trained results parameters 1280 720 batches 4 epochs 20 lr 1e-4 (has evaluation / model)
  - `obj_det_no_gpu/results3/`: Third trained results parameters 1280 720 batches 4 epochs 40 lr 1e-3 (has evaluation / model)
  - `obj_det_no_gpu/results4/`: Third trained results parameters 1920 1080 batches 2 epochs 200 lr 1e-3 (has evaluation / model)
  - `obj_det_no_gpu/train.py/`: Juypter notebook, training object detection for waste with modal lab code implementation
  - `obj_det_no_gpu/plastic.yaml/`: Yaml file to define paths to validation and training data / classes

- `obj_det_no_gpu/`: Directory for our object detection model and results (trained w/ gpu, using Modal Labs)
  - `obj_det_no_gpu/datasets/`: Cleaned / Modified datasets for our training (e.g., Hugging Face dataset, manually annotated data).
  - `obj_det_no_gpu/training_results/`: The best.pt, last.pt of the training results and additional quantitative results.
  - `obj_det_no_gpu/obj_detection.ipynb/`: Juypter notebook, training object detection for waste (parameters in notebook)
  - `obj_det_no_gpu/plastic.yaml/`: Yaml file to define paths to validation and training data / classes

- `requirements.txt`: Python package dependencies.
- `README.md`: This file.

  This project consists of object detection and class identification of waste in rivers.