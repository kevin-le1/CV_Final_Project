# Water Pollutant Detection and Classification

This repository contains the code and resources for the research project "Machine Learning for Water Pollutant Detection and Classification." The project aims to develop a robust computer vision system leveraging machine learning techniques to accurately detect and classify various types of water pollutants from images and videos captured in rivers, lakes, and coastal waters.

## Project Overview

The system employs a two-stage approach:

1. **Object Detection**: Localize pollutants within images or video frames using state-of-the-art object detection models like Faster R-CNN and YOLO.
2. **Multi-class Classification**: Categorize the detected objects into predefined classes (e.g., plastic, aluminum, oil, sewage) using deep neural network architectures like ResNet and EfficientNet.

The project explores strategies to handle challenges such as varying lighting conditions, water turbidity, occlusions, and class imbalances, employing techniques like data augmentation, transfer learning, ensemble methods, and attention mechanisms.

## Repository Structure

- `data/`: Directory for storing datasets (e.g., Hugging Face dataset, manually annotated data).
- `models/`: Trained model weights and configurations.
- `src/`: Source code for the project.
  - `src/detection/`: Object detection module.
  - `src/classification/`: Multi-class classification module.
  - `src/utils/`: Utility functions and helpers.
- `notebooks/`: Jupyter Notebooks for experimentation and analysis.
- `results/`: Directory for storing evaluation results, visualizations, and performance metrics.
- `requirements.txt`: Python package dependencies.
- `README.md`: This file.
  This project consists of object detection and class identification of waste in rivers.

Object detection uses yolov8 architecture!

CNN uses \_\_\_ architecture!

More details in respective directory!
