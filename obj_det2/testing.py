import modal
import yaml
import os

# kevin will test this to optimize object detection training to potentially get better results in the future

# Initialize the Modal app instance
app = modal.App()

# Create the image
image = modal.Image.debian_slim().apt_install("libgl1-mesa-glx", "libglib2.0-0").pip_install("ultralytics", "pyyaml", "datasets", "torchvision", "imageai", "numpy", "matplotlib", "pandas", "scikit-image", "ipykernel", "tensorflow", "opencv-python==4.8.0.74")


'''
from datasets import load_dataset

dataset = load_dataset('Kili/plastic_in_river', num_proc=6)

print(dataset)

import os

# only creating datasets for train and validation not test
os.makedirs('datasets/images/train', exist_ok=True)
os.makedirs('datasets/images/validation', exist_ok=True)

os.makedirs('datasets/labels/train', exist_ok=True)
os.makedirs('datasets/labels/validation', exist_ok=True)


def create_dataset(data, split):
  data = data[split]

  print(f'Running for {split} split...')

  for idx, sample in enumerate(data):
    image = sample['image']
    labels = sample['litter']['label']
    bboxes = sample['litter']['bbox']
    targets = []
    
    # creating the label txt files
    for label, bbox in zip(labels, bboxes):
      targets.append(f'{label} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}')
      
    with open(f'datasets/labels/{split}/{idx}.txt', 'w') as f:
      for target in targets:
        f.write(target + '\n')
        
    # saving image to png
    image.save(f'datasets/images/{split}/{idx}.png')

# create_dataset(dataset, 'train')
# create_dataset(dataset, 'validation')
'''

@app.function(gpu="A100-40GB",  image=image)
def test(yaml_data):
    from ultralytics import YOLO
    
    model = YOLO('yolov8m.pt')  # yolov8 architecture

    
    model.train(
        data=yaml_data,  # this plastic.yaml is the config file for object detection
        epochs=20,  # relatively low for now just for testing
        imgsz=(1280, 720),  # width, height
        batch=4,
        optimizer='Adam',
        lr0=1e-3,
        resume = True,
        device = '0'
    )

@app.local_entrypoint()
def main():
    # Load the YAML data from the file
    with open('plastic.yaml', 'r') as file:
        yaml_data = yaml.load(file, Loader=yaml.FullLoader)

    # Convert yaml_data to a string
    yaml_str = yaml.dump(yaml_data)

    test.remote(yaml_str)

if __name__ == "__main__":
    main()