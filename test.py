from datasets import load_dataset, load_from_disk
from PIL import Image
import pandas as pd
import numpy as np
import cv2
# used when testing most above not needed rn!

# loading the dataset
dataset = load_dataset("kili-technology/plastic_in_river") # use this
# dataset.save_to_disk("plastic_in_river_dataset") # TO SAVE DATA

# printing the lists of the datasets (3 maps)
print(list(dataset.keys()))
print(list(dataset['train'][0].keys()))

print(list(dataset['train'].shape))
print(list(dataset['train'][3404]))

# prints the PIL for testing
print(dataset['train'][0]['image'])

# NOTES PLEASE USE THE "PLASTIC_IN_RIVER_DATASET" NOT LOAD DATASET! in our final we are not going to use load_dataset 
# we will manipulate this load from disk dataset when we want to ! we have to parse thru it to relabel issue later ! think
# abt it later!

# saved_dataset = load_from_disk("plastic_in_river_dataset") # dont need this lied!

# converts PIL image to numpy array
# pil_image = saved_dataset['test'][0]['image']

# I = np.asarray(pil_image) # need to do this for every image test

# kevin will soon manipulate data

# display the image data from each pil image
#pil_image.show()


# iterate through the dataset, convert PIL images to NumPy arrays, and replace the image data
for split_name in dataset.keys(): # pass in dataset
    for i, example in enumerate(dataset[split_name]):
        # Convert PIL image to NumPy array
        pil_image = example['image']
        np_image = np.asarray(pil_image)

        # Replace PIL image with NumPy array
        print(np_image)
        dataset[split_name][i]['image'] = np_image

dataset.set_format("numpy")
# Save the modified dataset
dataset.save_to_disk("saved_dataset")