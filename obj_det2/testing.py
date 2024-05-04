import os
import pathlib
import yaml
from datasets import load_dataset
from modal import App, Image, Volume
import torch
from hub_sdk import HUBClient
import matplotlib.pyplot as plt


app = App()

# Define the image with required dependencies
image = (
    Image.debian_slim()
    .apt_install("libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        "ultralytics",
        "pyyaml",
        "datasets",
        "torchvision",
        "imageai",
        "numpy",
        "matplotlib",
        "pandas",
        "scikit-image",
        "ipykernel",
        "tensorflow",
        "opencv-python==4.8.0.74",
        "hub-sdk",
        "tblib",
        "dill",
    )
)

# Create a volume
volume = Volume.from_name("my-persisted-volume", create_if_missing=True)

# Load dataset
dataset = load_dataset('Kili/plastic_in_river')

# Function to create dataset
@app.function(volumes={"/root/datasets": volume}, image=image, timeout=86400)
def create_dataset(idx, sample, split):
    os.makedirs(f"/root/datasets/images/{split}", exist_ok=True)
    os.makedirs(f"/root/datasets/labels/{split}", exist_ok=True)

    print(f"Running for {split} split...")

    image = sample["image"]
    labels = sample["litter"]["label"]
    bboxes = sample["litter"]["bbox"]
    targets = []

    # Creating the label txt files
    for label, bbox in zip(labels, bboxes):
        targets.append(f"{label} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}")

    with open(f"/root/datasets/labels/{split}/{idx}.txt", "w") as f:
        for target in targets:
            f.write(target + "\n")

    # Saving image to png
    image.save(f"/root/datasets/images/{split}/{idx}.png")
    volume.commit()


# Function to test YOLO model
@app.function(
    gpu="H100", image=image, timeout=86400, volumes={"/root/datasets": volume}
)
def test():

    volume.reload()
    from ultralytics import YOLO
    import matplotlib.pyplot as plt
    import tblib
    from PIL import Image

    model = YOLO("yolov8m.pt")  # yolov8 architecture

    # Your dataset configuration as a dictionary
    dataset_config = {
        "path": ".",
        "train": "images/train",
        "val": "images/test",
        "names": {
            0: "PLASTIC_BAG",
            1: "PLASTIC_BOTTLE",
            2: "OTHER_PLASTIC_WASTE",
            3: "NOT_PLASTIC_WASTE",
        },
    }

    # Convert the dictionary to a YAML string
    yaml_str = yaml.dump(dataset_config)

    # Save the YAML string to a file
    with open("plastic.yaml", "w") as file:
        file.write(yaml_str)

        # Making the code device-agnostic
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Transferring the model to a CUDA enabled GPU
    model = model.to(device)
    save_dir = f'/root/datasets/results2'
    os.makedirs(save_dir, exist_ok=True)

    model.train(
        data='plastic.yaml',  # this plastic.yaml is the config file for object detection
        epochs=40,  # relatively low for now just for testing
        imgsz=(1920, 1080),  # width, height
        batch=4,
        optimizer='Adam',
        lr0=1e-3,
        device=0,
        project=save_dir,
    )
    volume.commit()
    
    torch.save(model.state_dict(), f'/root/datasets/trained10pt2epoch.pt')
    volume.commit()
    
    model.save('yolov810epochtest.pt')
    
    volume.commit()


# Main entry point
@app.local_entrypoint()
def main():
    
    train_dataset = dataset['train']
    for idx, sample in enumerate(train_dataset):
        create_dataset.remote(idx, sample, "train")

    test_dataset = dataset["test"]
    for idx, sample in enumerate(test_dataset):
        create_dataset.remote(idx, sample, 'test')
    
    test.remote()


if __name__ == "__main__":
    main()
