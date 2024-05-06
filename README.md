# Water Pollutant Detection and Classification

This repository contains the code and resources for the research project "Machine Learning for Water Pollutant Detection and Classification". The project aims to develop a robust computer vision system leveraging machine learning techniques to accurately detect and classify various types of water pollutants from images and videos captured in rivers, lakes, and coastal waters.

## Project Overview

The system employs a two-stage approach:

1. **Multi-class Classification**: Categorize the images into predefined classes (e.g., clean water, spils (oil), unclean water (trash in water)) using deep neural network architectures, inspired from ResNet.
2. **Object Detection**: Localize pollutants within images or video frames using state-of-the-art object detection models such as YOLOv8 architecture and adam as an optimizer.

The project explores strategies to handle challenges such as varying lighting conditions, water turbidity, occlusions, and class imbalances, employing techniques like data augmentation, transfer learning, ensemble methods, and attention mechanisms.

## Repository Structure

### CNN

- `cnn/`: Directory for our CNN model and results
  - `cnn/data.ipynb`: Jupyter notebook for dataset creation and preparation
  - `cnn/model.py`: Class representing the PyTorch model, using residual layers a la ResNet with added Dropout layers
  - `cnn/preprocess.py`: Functions to handle image normalization and PyTorch DataLoader creation; prepared test, validation, and train sets
  - `cnn/train.py`: Main train script
  - `cnn/eval.py`: Takes a model and outputs a confusion matrix on the test set; the `--test` runs the evaluation on the test set and the `--dsinfo` outputs information about inter class distribution in the dataset
  - `cnn/oil_images.py`: Script to scrape Google Images for the top 20 images ranked under a search parameter; used to gather images for the dataset on oil spills
  - `cnn/wandb_data.py`: Script to send .csv data to Weights and Biases (wandb) for visualization and analysis; reads a given `.csv` file using pandas and sends data row by row, with cell mutation and row omission (if necessary)
  - `cnn/oil/`: Directory containing subfolders of results of `oil_images.py`; these search results are ingressed by the `data.ipynb` file in the creation of the dataset

How to run:

1. Install the necessary dependencies using `requirements.txt`
2. Run the cells in `data.ipynb` to download the dataset and organize it into train/validation/test subdirectories with class label (0-2) folder structures
3. Configure hyperparameters in `train.py`, including the Weights and Biases (wandb) API key for training visualization
4. Run `python train.py` to initiate training of the model
   1. Optionally, precompute dataset mean and standard deviation using `python preprocess.py` and replace `ds_mean` and `ds_std_dev` in `load_data` in `preprocess.py`
5. Optionally, Use `python eval.py` to create a confusion matrix for the finished model

### Object Detection

<br />

- `obj_det_no_gpu/`: Directory for our object detection model and results (trained w/o gpu data stored in modal labs)
  - `obj_det_no_gpu/results/`: First trained results parameters 1280 720 batches 4 epochs 10 lr 1e-4 (has evaluation / model)
  - `obj_det_no_gpu/results2/`: Second trained results parameters 1280 720 batches 4 epochs 20 lr 1e-4 (has evaluation / model)
  - `obj_det_no_gpu/results3/`: Third trained results parameters 1280 720 batches 4 epochs 40 lr 1e-3 (has evaluation / model)
  - `obj_det_no_gpu/results4/`: Third trained results parameters 1920 1080 batches 2 epochs 200 lr 1e-3 (has evaluation / model)
  - `obj_det_no_gpu/train.py/`: Juypter notebook, training object detection for waste with modal lab code implementation

<br />
How to run:
<br />

After installing all of the dependencies, cd into `obj_det_no_gpu/` and run the following command:
  - `modal run train.py`
<br />

To obtain the results, run the following command
  - `modal volume get my-persisted-volume (path) (dest)`

<br />

- `obj_det_gpu/`: Directory for our object detection model and results (trained w/ gpu, using Modal Labs)
  - `obj_det_gpu/datasets/`: Cleaned / Modified datasets for our training (e.g., Hugging Face dataset, manually annotated data).
  - `obj_det_gpu/training_results/`: The best.pt, last.pt of the training results and additional quantitative results.
  - `obj_det_gpu/obj_detection.ipynb/`: Juypter notebook, training object detection for waste (parameters in notebook)
  - `obj_det_gpu/plastic.yaml/`: Yaml file to define paths to validation and training data / classes

<br />
How to run:
<br />
After installing all of the dependencies, run each cell in `obj_det_no_gpu/obj_detection.ipynb/`.

<br />

- `requirements.txt`: Python package dependencies.
- `README.md`: This file.

  This project consists of object detection and class identification of waste in rivers.