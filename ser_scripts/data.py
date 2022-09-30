# data.py

from torch.utils.data import DataLoader
from torchvision import datasets
from pathlib import Path   
from ser_scripts.transforms import transform_torch 

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"

def create_dataloaders(batch_size):

    # dataloaders
    training_dataloader = DataLoader(
        datasets.MNIST(root="../data", download=True, train=True, transform=transform_torch()),
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
    )

    validation_dataloader = DataLoader(
        datasets.MNIST(root=DATA_DIR, download=True, train=False, transform=transform_torch()),
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
    )

    return(training_dataloader, validation_dataloader)