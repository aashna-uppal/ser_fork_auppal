from torch import optim
import torch
import torch.nn.functional as F
from visdom import Visdom
from ser.model import Net
import numpy as np
import ser.utils
import json

#Create global plotter variable (as per the visdom tutorial)
global plotter
plotter = ser.utils.VisdomLinePlotter(env_name='ser_fork_auppal')


def train(run_path, params, train_dataloader, val_dataloader, device):
    # setup model
    model = Net().to(device)

    # setup params
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)

    # train
    for epoch in range(params.epochs):
        _train_batch(model, train_dataloader, optimizer, epoch, device)
        _val_batch(model, val_dataloader, device, epoch, run_path)

    # save model and save model params
    torch.save(model, run_path / "model.pt")


def _train_batch(model, dataloader, optimizer, epoch, device):
    for i, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)
        model.train()
        optimizer.zero_grad()
        output = model(images)
        loss = F.nll_loss(output, labels)
        loss.backward()
        optimizer.step()
        print(
            f"Train Epoch: {epoch} | Batch: {i}/{len(dataloader)} "
            f"| Loss: {loss.item():.4f}"
        )

        #PLOT
        plotter.plot(var_name = 'Training Loss', split_name = 'Train Loss', title_name = "Training Loss (per Batch, not Epoch)", x = i+(len(dataloader)*epoch), x_name = "Batch", y = loss.item())


@torch.no_grad()
def _val_batch(model, dataloader, device, epoch, run_path):
    val_loss = 0
    correct = 0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        model.eval()
        output = model(images)
        val_loss += F.nll_loss(output, labels, reduction="sum").item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(labels.view_as(pred)).sum().item()
    val_loss /= len(dataloader.dataset)
    accuracy = correct / len(dataloader.dataset)

    #PRINT
    print(f"Val Epoch: {epoch} | Avg Loss: {val_loss:.4f} | Accuracy: {accuracy}")

    #SAVE
    results = {"Val Epoch": epoch, "Avg Loss": val_loss, "Accuracy": accuracy}
    with open(run_path / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    #PLOT
    plotter.plot(var_name = 'Validation Loss & Accuracy', split_name = 'Val Loss', title_name = "Validation Loss & Accuracy", x = epoch, x_name = "Epoch", y = val_loss)
    plotter.plot(var_name = 'Validation Loss & Accuracy', split_name = 'Val Accuracy', title_name = "Validation Loss & Accuracy", x = epoch, x_name = "Epoch", y = accuracy)
