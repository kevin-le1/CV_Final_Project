import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import ConfusionMatrixDisplay

from model import ResNet

from preprocess import load_data

##
NUM_CLASSES = 3
BATCH_SIZE = 128
DROPOUT_RATE = 0.5

ROOT_IMG_DIR = "datasets/images"
MODEL_DIR = "models"

LOAD_MODEL_FILE = f"{MODEL_DIR}/trained_model_87_2024-05-05_19:27:41_loss_0.6552832722663879_acc_69.07894736842105.pt"
##


def test(
    model: ResNet,
    device: torch.device,
    test_loader: DataLoader,
    criterion: nn.CrossEntropyLoss,
    test: bool = True,
):
    """
    Evaluate test accuracy of model
    """
    correct = 0
    loss = 0
    all_predictions = []
    all_labels = []

    total_mini_batches = len(test_loader.dataset) // BATCH_SIZE + 1  # type: ignore

    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            # compute
            outputs = model(inputs)
            # calculate running loss
            loss += criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    size = len(test_loader.dataset)  # type: ignore
    loss /= total_mini_batches
    accuracy = correct / size

    # generate Confusion Matrix
    # cm = confusion_matrix(all_labels, all_predictions)
    disp = ConfusionMatrixDisplay.from_predictions(
        all_labels, all_predictions, normalize="true"
    )
    disp.plot().figure_.savefig("confusion_matrix1.png")

    eval_type = "Test" if test else "Val"

    print(
        f"[{eval_type} eval]\tAverage loss: {loss:.4f}\tAccuracy: {correct}/{size} ({accuracy:.2%})"
    )


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # initialize model
    layer_depths = [2, 2, 2, 2]
    model = ResNet(layer_depths, NUM_CLASSES, DROPOUT_RATE)
    model = model.to(device)  # move to GPU if available

    # define loss function and optimizer
    criterion = nn.CrossEntropyLoss()

    if LOAD_MODEL_FILE is not None:
        model.load_state_dict(torch.load(LOAD_MODEL_FILE))

    # load dataset
    train_loader, _ = load_data(ROOT_IMG_DIR, BATCH_SIZE, shuffle=True)

    test_loader = load_data(ROOT_IMG_DIR, BATCH_SIZE, shuffle=False, test=True)
    # from collections import Counter

    # print(dict(Counter(train_loader.dataset.targets)))

    test(model, device, train_loader, criterion)  # type: ignore


if __name__ == "__main__":
    main()
