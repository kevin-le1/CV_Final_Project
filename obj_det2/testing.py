import modal

# Initialize the Modal app instance
app = modal.App()

# Create the image
image = modal.Image.debian_slim().pip_install("ultralytics", "datasets", "torchvision", "imageai", "numpy", "matplotlib", "pandas", "scikit-image", "ipykernel", "tensorflow", "opencv-python==4.8.0.74")


@app.function(gpu="A100-40GB",  image=image)
def test():
    from ultralytics import YOLO
    
    model = YOLO('yolov8m.pt')  # yolov8 architecture
    
    model.train(
        data='plastic.yaml',  # this plastic.yaml is the config file for object detection
        epochs=1,  # relatively low for now just for testing
        imgsz=(1280, 720),  # width, height
        batch=4,
        optimizer='Adam',
        lr0=1e-3,
    )

@app.local_entrypoint()
def main():
    test.remote()

if __name__ == "__main__":
    main()