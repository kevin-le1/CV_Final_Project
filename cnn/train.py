"""
Main train script for ResNet CNN
"""

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim

from model import ResNet
from tqdm import tqdm
import wandb

from datetime import datetime, timedelta


from preprocess import load_data

# ---- hyperparameters and other configs
NUM_CLASSES = 3  # CLEAN, OIL, TRASH
NUM_EPOCHS = 200
BATCH_SIZE = 128  # note that long run used batch size 64
LEARNING_RATE = 0.0004
MAX_LEARNING_RATE = 0.01
LR_PATIENCE = 2
LR_FACTOR = 0.5
WEIGHT_DECAY = 0.0001
DROPOUT_RATE = 0.5
LOG_INTERVAL = 5  # minibatch interval not epochs
API_KEY = None

ROOT_IMG_DIR = "datasets/images"
MODEL_DIR = "models"

RESUME = False
RESUME_ID = None
# LOAD_MODEL_FILE = f"{MODEL_DIR}/"
LOAD_MODEL_FILE = None
# ----

wandb_enabled = bool(API_KEY is not None)


def get_lr(optimizer: optim.Optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def save(model: ResNet, epoch: int, test_loss: float, test_acc: float):
    """
    Save model weights to persistent volume
    """
    now = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    file_name = f"trained_model_{epoch}_{now}_loss_{test_loss}_acc_{test_acc}.pt"
    alt_name = f"trained_model_{epoch}_loss_{test_loss}_acc_{test_acc}.pt"
    # save the trained model to a file
    torch.save(model.state_dict(), f"{MODEL_DIR}/{file_name}")
    if wandb_enabled:
        artifact = wandb.Artifact(alt_name, type="model")
        artifact.add_file(f"{MODEL_DIR}/{file_name}")
        wandb.log_artifact(artifact)


def test(
    model: ResNet,
    device: torch.device,
    test_loader: DataLoader,
    criterion: nn.CrossEntropyLoss,
    show_img: bool = True,
    test: bool = True,
):
    """
    Evaluate test accuracy of model
    """
    correct = 0
    loss = 0
    example_imgs = []  # for wandb
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

            example_imgs.append(
                wandb.Image(
                    inputs[0],  # grab first image
                    caption=f"Pred: {predicted[0]} Truth: {labels[0]}",
                )
            )

    size = len(test_loader.dataset)  # type: ignore
    # TODO: change to total_mini_batches
    loss /= size
    accuracy = correct / size

    eval_type = "Test" if test else "Val"

    print(
        f"[{eval_type} eval]\tAverage loss: {loss:.4f}\tAccuracy: {correct}/{size} ({accuracy:.2%})"
    )
    if wandb_enabled:
        log_dict = {
            f"{eval_type} Accuracy": 100.0 * correct / size,
            f"{eval_type} Loss": loss,
        }
        if show_img:
            log_dict["Examples"] = example_imgs

        wandb.log(log_dict)

    return {
        f"{eval_type} Accuracy": 100.0 * correct / size,
        f"{eval_type} Loss": loss,
    }


def train(
    model: ResNet,
    device: torch.device,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.CrossEntropyLoss,
    epoch: int,
):
    """
    Train CNN classification model for one epoch
    """
    model.train()  # tell pytorch we're training
    total_elapsed = timedelta(0)
    total_mini_batches = len(train_loader.dataset) // BATCH_SIZE + 1  # type: ignore
    correct = 0
    total_loss = 0
    for i, (inputs, labels) in enumerate(train_loader, 0):
        start_time = datetime.now()

        inputs, labels = inputs.to(device), labels.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        # eval
        total_loss += loss
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
        #
        loss.backward()
        optimizer.step()

        # analytics
        end_time = datetime.now()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # time manipulation
        elapsed_time = end_time - start_time
        total_elapsed += elapsed_time
        avg_elapsed = total_elapsed / (i + 1)
        time_remaining = avg_elapsed * (total_mini_batches - i - 1)
        est_completion_time = datetime.now() + time_remaining

        if (i + 1) % LOG_INTERVAL == 0:
            print(
                f"[{now} - Epoch {epoch}] {(i + 1) * len(inputs)}/{len(train_loader.dataset)}\tLoss: {loss.item():.4f}"  # type: ignore
            )
            print(
                f"[{now} - Epoch {epoch}] Elapsed time {elapsed_time}\tTotal time {total_elapsed}\tTime remaining {avg_elapsed * (total_mini_batches - i - 1)}@{est_completion_time.strftime('%H:%M:%S')}"
            )

        # log batch train loss to wandb
        if wandb_enabled:
            wandb.log({"Minibatch Train Loss": loss.item()})

    size = len(train_loader.dataset)  # type: ignore
    accuracy = correct / size
    avg_loss = total_loss / total_mini_batches
    print(
        f"[Train eval]\tAverage loss: {avg_loss:.4f}\tAccuracy: {correct}/{size} ({accuracy:.2%})"
    )

    if wandb_enabled:
        # TODO: multiply accuracy by 100.0 here
        wandb.log({"Average Train Loss": avg_loss, "Train Accuracy": accuracy})


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # enable wandb for analytics
    if wandb_enabled:
        wandb.login(key=API_KEY)
        wandb.init(
            project="cv-final-resnet-training",
            resume=RESUME,
            id=RESUME_ID,
        )
        wandb.config.update(
            {"lr": LEARNING_RATE},
            allow_val_change=True,
        )
        # wandb.config.lr = LEARNING_RATE
        wandb.config.lr_patience = LR_PATIENCE
        wandb.config.lr_factor = LR_FACTOR
        wandb.config.wd = WEIGHT_DECAY
        wandb.config.batch_size = BATCH_SIZE
        wandb.config.dropout = DROPOUT_RATE

    # According to Table 1 in the ResNet paper for a 34-layer model
    # layer_depths = [3, 4, 6, 3]
    # According to Table 1 in the ResNet paper for a 18-layer model
    layer_depths = [2, 2, 2, 2]
    # initialize model
    model = ResNet(layer_depths, NUM_CLASSES, DROPOUT_RATE)
    # move to GPU if available
    model = model.to(device)

    # define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # TODO: load model file here if provided
    if wandb_enabled and wandb.run.resumed:  # type: ignore
        if LOAD_MODEL_FILE is not None:
            model.load_state_dict(torch.load(LOAD_MODEL_FILE))

    # load dataset
    train_loader, val_loader = load_data(ROOT_IMG_DIR, BATCH_SIZE, shuffle=True)

    test_loader = load_data(ROOT_IMG_DIR, BATCH_SIZE, shuffle=False, test=True)

    # sched = torch.optim.lr_scheduler.OneCycleLR(
    #     optimizer,
    #     MAX_LEARNING_RATE,
    #     epochs=NUM_EPOCHS,
    #     steps_per_epoch=len(train_loader),
    # )
    # note switched to ReduceLROnPlateau around step 1600 on wandb
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "max", factor=LR_FACTOR, patience=LR_PATIENCE
    )

    # automatically track gradients with wandb
    if wandb_enabled:
        wandb.watch(model, criterion=criterion, log_freq=1)

    best_test_acc = 0

    for epoch in tqdm(range(NUM_EPOCHS), desc="Training model"):
        train(model, device, train_loader, optimizer, criterion, epoch)  # type: ignore
        test_res = test(model, device, test_loader, criterion)  # type: ignore
        val_res = test(model, device, val_loader, criterion, show_img=False, test=False)  # type: ignore
        if test_res["Test Accuracy"] > best_test_acc:
            best_test_acc = test_res["Test Accuracy"]
            print(f"[ALERT] new highest test accuracy at epoch {epoch}")
        save(model, epoch, test_res["Test Loss"], test_res["Test Accuracy"])
        sched.step(test_res["Test Accuracy"])
        print(
            f"[Epoch {epoch} END]\tlr: {get_lr(optimizer)}\ttest loss: {test_res['Test Loss']}\ttest acc: {test_res['Test Accuracy']}\tval loss: {val_res['Val Loss']}\tval acc: {val_res['Val Accuracy']}"
        )

        # track learning rate
        if wandb_enabled:
            wandb.log({"Learning rate": get_lr(optimizer)})

    if wandb_enabled:
        wandb.finish()


if __name__ == "__main__":
    main()
