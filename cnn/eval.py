import torch
from torch.utils.data import DataLoader

from sklearn.metrics import ConfusionMatrixDisplay, classification_report

from collections import Counter
import argparse

from model import ResNet
from preprocess import load_data

##
NUM_CLASSES = 3
BATCH_SIZE = 128
DROPOUT_RATE = 0.5

ROOT_IMG_DIR = "datasets/images"
MODEL_DIR = "models"

# LOAD_MODEL_FILE = f"{MODEL_DIR}/"
LOAD_MODEL_FILE = None
MODEL_NUM = 0
##


def test(
    model: ResNet,
    device: torch.device,
    test_loader: DataLoader,
    save_name: str = f"{MODEL_NUM}",
):
    """
    Evaluate test accuracy of model
    """
    correct = 0
    all_predictions = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            # compute
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    size = len(test_loader.dataset)  # type: ignore
    accuracy = correct / size

    # print(accuracy, correct, size)

    print(classification_report(all_labels, all_predictions))

    # generate Confusion Matrix
    cm_disp = ConfusionMatrixDisplay.from_predictions(
        all_labels, all_predictions, normalize="true"
    )
    cm_disp.plot().figure_.savefig(f"{save_name}_confusion_matrix.png")


def main():
    # args
    parser = argparse.ArgumentParser(description="Eval script arguments")
    parser.add_argument(
        "--test", action="store_true", help="Enable test mode (default: False)"
    )
    parser.add_argument(
        "--dsinfo",
        action="store_true",
        help="Print info about the train and test datasets (default: False)",
    )
    args = parser.parse_args()
    #

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load dataset
    train_loader, _ = load_data(ROOT_IMG_DIR, BATCH_SIZE, shuffle=True)

    test_loader = load_data(ROOT_IMG_DIR, BATCH_SIZE, shuffle=False, test=True)

    if args.dsinfo:
        print("Train class distribution", dict(Counter(train_loader.dataset.targets)))  # type: ignore
        print("Test class distribution", dict(Counter(test_loader.dataset.targets)))  # type: ignore

    if LOAD_MODEL_FILE is None:
        return

    # initialize model
    layer_depths = [2, 2, 2, 2]
    model = ResNet(layer_depths, NUM_CLASSES, DROPOUT_RATE)
    model.load_state_dict(torch.load(LOAD_MODEL_FILE))
    model = model.to(device)  # move to GPU if available

    if not args.test:
        test(model, device, train_loader, f"{MODEL_NUM}_train")
    else:
        test(model, device, test_loader, f"{MODEL_NUM}_test")  # type: ignore


if __name__ == "__main__":
    main()
