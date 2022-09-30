# train.py

import torch
from torch import optim
import torch.nn.functional as F
import torch.nn as nn
from ser_scripts.model import model_load
from ser_scripts.data import create_dataloaders

def train_model(learning_rate, epochs, batch_size):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for epoch in range(epochs):

        # train ---------------------------------------------
        for i, (images, labels) in enumerate(create_dataloaders(batch_size)[0]):
            images, labels = images.to(device), labels.to(device)
            model_load(learning_rate)[0].train()
            model_load(learning_rate)[1].zero_grad()
            output = model_load(learning_rate)[0](images)
            loss = F.nll_loss(output, labels)
            loss.backward()
            model_load(learning_rate)[1].step()
            print(
                f"Train Epoch: {epoch} | Batch: {i}/{len(create_dataloaders(batch_size)[0])} "
                f"| Loss: {loss.item():.4f}"
            )

        # validate ---------------------------------------------
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for images, labels in create_dataloaders(batch_size)[1]:
                images, labels = images.to(device), labels.to(device)
                model_load(learning_rate)[0].eval()
                output = model_load(learning_rate)[0](images)
                val_loss += F.nll_loss(output, labels, reduction="sum").item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(labels.view_as(pred)).sum().item()
            val_loss /= len(create_dataloaders(batch_size)[1].dataset)
            val_acc = correct / len(create_dataloaders(batch_size)[1].dataset)

        print(f"Val Epoch: {epoch} | Avg Loss: {val_loss:.4f} | Accuracy: {val_acc}")