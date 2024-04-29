import modal
import os
from modal._utils.grpc_utils import retry_transient_errors
import yaml

# kevin will test this to optimize object detection training to potentially get better results in the future

# Initialize the Modal app instance
app = modal.App()

# Create the image
image = modal.Image.debian_slim().apt_install("libgl1-mesa-glx", "libglib2.0-0").pip_install("ultralytics", "pyyaml", "datasets", "torchvision", "imageai", "numpy", "matplotlib", "pandas", "scikit-image", "ipykernel", "tensorflow", "opencv-python==4.8.0.74")


import pathlib

app = modal.App()  # Note: prior to April 2024, "app" was called "stub"

volume = modal.Volume

p = pathlib.Path("/root/datasets")

from datasets import load_dataset

dataset = load_dataset('Kili/plastic_in_river', num_proc=6)

@app.function(volumes={"/root/datasets": volume}, image = image, timeout=86400)
def create_dataset(data, split):
    
    os.makedirs('root/datasets/images/train', exist_ok=True)
    os.makedirs('root/datasets/images/validation', exist_ok=True)

    os.makedirs('root/datasets/labels/train', exist_ok=True)
    os.makedirs('root/datasets/labels/validation', exist_ok=True)
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

        with open(f'/root/datasets/labels/{split}/{idx}.txt', 'w') as f:
            for target in targets:
                f.write(target + '\n')

        # saving image to png
        image.save(f'/root/datasets/images/{split}/{idx}.png')



@app.function(gpu="A100-40GB",  image=image, timeout=86400)
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
    )

@app.local_entrypoint()
def main():
    create_dataset.remote(dataset, 'train')
    create_dataset.remote(dataset, 'test')
    
    with open("data.yaml", 'r') as stream:
        try:
            data = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    
    test.remote(data)

if __name__ == "__main__":
    main()