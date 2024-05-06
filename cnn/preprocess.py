"""
Given a dataset of images, normalize the images and load them into dataloaders for training and inference
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Union
from PIL import Image
from tqdm import tqdm

import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy("file_system")

# ---- hyperparameters and other configs
IMG_HEIGHT = 224
IMG_WIDTH = 224

ROOT_IMG_DIR = "datasets/images"
TRAIN_IMG_DIR = f"{ROOT_IMG_DIR}/train"
VAL_IMG_DIR = f"{ROOT_IMG_DIR}/validation"
TEST_IMG_DIR = f"{ROOT_IMG_DIR}/test"
# ----


def compute_mean_std(
    dir: str,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    imgs = datasets.ImageFolder(dir)
    img_len = len(imgs)

    # init default mean/std
    mean = torch.zeros(3)
    std_dev = torch.zeros(3)

    # compute mean
    for _, rel_path in enumerate(
        tqdm(imgs.imgs, total=img_len, desc="Computing mean of dataset")
    ):
        image = Image.open(rel_path[0])
        image_tensor = transforms.ToTensor()(image)
        mean += torch.mean(image_tensor, dim=(1, 2))

    mean /= img_len

    # compute stddev (needs mean)
    for _, rel_path in enumerate(
        tqdm(imgs.imgs, total=img_len, desc="Computing stddev of dataset")
    ):
        image = Image.open(rel_path[0])
        image_tensor = transforms.ToTensor()(image)
        std_dev += torch.mean(
            (image_tensor - mean.unsqueeze(1).unsqueeze(2)) ** 2, dim=(1, 2)
        )

    std_dev = torch.sqrt(std_dev / img_len)

    return mean.tolist(), std_dev.tolist()  # type: ignore


def load_data(
    directory: str, batch_size: int = 64, shuffle: bool = True, test: bool = False
) -> Union[DataLoader, tuple[DataLoader, DataLoader]]:
    # preprocess dataset

    # for the sake of efficiency, compute once and then use known good values
    # TODO: recompute with oil spill images added (takes ~5min locally)
    # ds_mean, ds_std_dev = compute_mean_std(ROOT_IMG_DIR)

    ds_mean = (0.5442656874656677, 0.5609508156776428, 0.5326545238494873)
    ds_std_dev = (0.2252025306224823, 0.2230863869190216, 0.25379157066345215)

    # define photo preprocess transformation
    train_transform = transforms.Compose(
        [
            transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
            transforms.RandomHorizontalFlip(p=0.5),  # randomly flip and rotate
            transforms.ColorJitter(0.3, 0.4, 0.4, 0.2),  # modify color
            transforms.ToTensor(),
            transforms.Normalize(ds_mean, ds_std_dev),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
            transforms.ToTensor(),
            transforms.Normalize(ds_mean, ds_std_dev),
        ]
    )

    # load dataset
    if test:
        test_imgs = datasets.ImageFolder(TEST_IMG_DIR, transform=test_transform)
        test_loader = DataLoader(test_imgs, batch_size, shuffle=False, num_workers=2)
        return test_loader

    train_imgs = datasets.ImageFolder(TRAIN_IMG_DIR, transform=train_transform)
    train_loader = DataLoader(
        train_imgs,
        batch_size,
        shuffle,
        num_workers=4,
        prefetch_factor=10,
        persistent_workers=True,
    )

    val_imgs = datasets.ImageFolder(VAL_IMG_DIR, transform=test_transform)
    val_loader = DataLoader(val_imgs, batch_size, shuffle, num_workers=2)

    return (train_loader, val_loader)


if __name__ == "__main__":
    print(compute_mean_std(ROOT_IMG_DIR))
