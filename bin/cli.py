#Disclaimer for Tom & Claudia: 
#I haven't done ML before and learned Python a handful of days before so I am totally misunderstanding what parts of this code do...
#But trying my best to complete the steps!

from pathlib import Path
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import typer

#Import refactored functions
from ser_scripts.transforms import transform_torch
from ser_scripts.model import model_load
from ser_scripts.data import create_dataloaders
from ser_scripts.train import train_model

main = typer.Typer()

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"


@main.command()
def train(

    name: str = typer.Option(
        ..., "-n", "--name", help="Name of experiment to save under."
    ),

    #Make all hyperparameters inputs via the cli using typer:
    #EPOCHS --------------------------------------------------
    epochs: int = typer.Option(
        ..., "-e", "--epochs", help="Add epochs."
    ),
    #BATCH SIZE ----------------------------------------------
    batch_size: int = typer.Option(
        ..., "-b", "--batch_size", help="Add batch size."
    ),
    #LEARNING RATE -------------------------------------------
    learning_rate: float = typer.Option(
        ..., "-l", "--learning_rate", help="Add learning rate."
    ),
):

    print(f"Running experiment {name}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load model and set up parameters
    model_load(learning_rate)

    # torch transforms
    transform_torch()

    # load training and validation data loader
    create_dataloaders(batch_size)

    # train model
    train_model(learning_rate, epochs, batch_size)


@main.command()
def infer():
    print("This is where the inference code will go")
