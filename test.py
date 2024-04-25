from datasets import load_dataset, load_from_disk
from PIL import Image
import pandas as pd
import numpy as np
# used when testing most above not needed rn!

# loading the dataset
dataset = load_dataset("kili-technology/plastic_in_river")

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

saved_dataset = load_from_disk("plastic_in_river_dataset")

# converts PIL image to numpy array
pil_image = saved_dataset['test'][0]['image']

# Display the image aggregate data from this each pil image
pil_image.show()

'''
dataset = pd.read_csv(
    hf_hub_download(repo_id=REPO_ID, filename=FILENAME, repo_type="dataset")
)

# load the image using PIL
pil_image = Image.open(dataset['train'][0]['image'], mode = 'r')

# converts PIL image to numpy array
open_cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

# display the image using OpenCV
cv2.imshow("Image", open_cv_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''