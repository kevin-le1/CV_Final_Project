from datasets import load_dataset
import cv2
from PIL import Image
import numpy as np


dataset = load_dataset("kili-technology/plastic_in_river")

print(dataset['train'][0]['image'])
print(list(dataset['train'][0].keys()))

# Load the image using PIL
pil_image = Image.open(dataset['train'][0]['image'], mode = 'r')

# Convert PIL image to numpy array
open_cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

# Display the image using OpenCV
cv2.imshow("Image", open_cv_image)
cv2.waitKey(0)
cv2.destroyAllWindows()