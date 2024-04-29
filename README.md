# Water Pollutant Detection and Classification

This repository contains the code and resources for the research project "Machine Learning for Water Pollutant Detection and Classification." The project aims to develop a robust computer vision system leveraging machine learning techniques to accurately detect and classify various types of water pollutants from images and videos captured in rivers, lakes, and coastal waters.

## Project Overview

The system employs a two-stage approach:

1. **Object Detection**: Localize pollutants within images or video frames using state-of-the-art object detection models such as YOLOv8 architecture and adam for momentum.
2. **Multi-class Classification**: Categorize the images into predefined classes (e.g., clean water, spils (oil), unclean water) using deep neural network architectures like ResNet and EfficientNet.

The project explores strategies to handle challenges such as varying lighting conditions, water turbidity, occlusions, and class imbalances, employing techniques like data augmentation, transfer learning, ensemble methods, and attention mechanisms.

## Repository Structure

- `cnn/`: Directory for our CNN model and results

- `evaluations/`: Quantitative and qualititative notebooks (includes images and writeups)

- `obj_detection/`: Directory for our object detection model and results
  - `obj_detection/datasets/`: Cleaned / Modified datasets for our training (e.g., Hugging Face dataset, manually annotated data).
  - `obj_detection/training_results/`: The best.pt, last.pt of the training results and additional quantitative results.

- `requirements.txt`: Python package dependencies.
- `README.md`: This file.

  This project consists of object detection and class identification of waste in rivers.