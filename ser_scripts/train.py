# train.py

import torch
from torch import optim
import torch.nn.functional as F
import torch.nn as nn
from ser_scripts.model import model_load
from ser_scripts.data import create_dataloaders

def train_model(learning_rate, epochs, batch_size):

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 32, 3, 1)
            self.conv2 = nn.Conv2d(32, 64, 3, 1)
            self.dropout1 = nn.Dropout(0.25)
            self.dropout2 = nn.Dropout(0.5)
            self.fc1 = nn.Linear(9216, 128)
            self.fc2 = nn.Linear(128, 10)

        def forward(self, x):
            x = self.conv1(x)
            x = F.relu(x)
            x = self.conv2(x)
            x = F.relu(x)
            x = F.max_pool2d(x, 2)
            x = self.dropout1(x)
            x = torch.flatten(x, 1)
            x = self.fc1(x)
            x = F.relu(x)
            x = self.dropout2(x)
            x = self.fc2(x)
            output = F.log_softmax(x, dim=1)
            return output

    # train
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for epoch in range(epochs):
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
            # validate
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

                print(
                    f"Val Epoch: {epoch} | Avg Loss: {val_loss:.4f} | Accuracy: {val_acc}"
                )