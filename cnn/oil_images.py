"""
Scrape Google images after a query argument to pad our dataset with more training/test/val images
"""

import requests
from bs4 import BeautifulSoup
import os
import urllib.request

# Set the search query
query = "oil_spill_bird_eye_view"

# Set the number of images to download
num_images = 20

# Set the directory to save the images
save_dir = f"oil/{query}"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Set the Google Images search URL
url = f"https://www.google.com/search?q={query}&tbm=isch"

# Send a request to the Google Images search page
response = requests.get(url)

# Parse the HTML content using BeautifulSoup
soup = BeautifulSoup(response.content, "html.parser")

# Find all the image links
image_links = [img.get("src") for img in soup.find_all("img")]

# Download the images
downloaded_images = 0
for i, link in enumerate(image_links):
    try:
        urllib.request.urlretrieve(link, os.path.join(save_dir, f"{i}.jpg"))
        print(f"Downloaded image {i+1}/{num_images}")
        downloaded_images += 1
        if downloaded_images >= num_images:
            break
    except:
        print(f"Error downloading image {i+1}/{num_images}")

print(f"Scraping complete! Downloaded {downloaded_images} images.")
