from datetime import datetime
from pathlib import Path
from xmlrpc.client import boolean

import typer
import torch
import git
import os
import json

from ser.train import train as run_train
from ser.constants import RESULTS_DIR
from ser.data import train_dataloader, val_dataloader, test_dataloader
from ser.infer import infer as run_infer
from ser.params import Params, save_params, load_params
from ser.transforms import transforms, normalize, flip
from ser.select_transforms import _select_test_image

main = typer.Typer()


@main.command()
def train(
    name: str = typer.Option(
        ..., "-n", "--name", help="Name of experiment to save under."
    ),
    epochs: int = typer.Option(
        5, "-e", "--epochs", help="Number of epochs to run for."
    ),
    batch_size: int = typer.Option(
        1000, "-b", "--batch-size", help="Batch size for dataloader."
    ),
    learning_rate: float = typer.Option(
        0.01, "-l", "--learning-rate", help="Learning rate for the model."
    ),
):
    """Run the training algorithm."""
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha

    # wraps the passed in parameters
    params = Params(name, epochs, batch_size, learning_rate, sha)

    # setup device to run on
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # setup run
    fmt = "%Y-%m-%dT%H-%M"
    timestamp = datetime.strftime(datetime.utcnow(), fmt)
    run_path = RESULTS_DIR / name / timestamp
    run_path.mkdir(parents=True, exist_ok=True)

    # Save parameters for the run
    save_params(run_path, params)

    # Train!
    run_train(
        run_path,
        params,
        train_dataloader(params.batch_size, transforms(normalize)),
        val_dataloader(params.batch_size, transforms(normalize)),
        device,
    )


@main.command()
def infer(
    run_path: Path = typer.Option(
        ..., "-p", "--path", help="Path to run from which you want to infer."
    ),
    label: int = typer.Option(
        6, "-l", "--label", help="Label of image to show to the model"
    ),
    flip_image: bool = typer.Option(
        False, "-f", "--flip_image", help="Flip image?"
    ),
    normalize_image: bool = typer.Option(
        False, "-n", "--normalize_image", help="Normalize image?"
    ),
):
    """Run the inference code"""
    params = load_params(run_path)
    model = torch.load(run_path / "model.pt")
    image = _select_test_image(label, flip_image, normalize_image)
    run_infer(params, model, image, label)

@main.command()
def model_load():
    # Getting the current work directory (cwd)
    thisdir = os.getcwd()

    # r=root, d=directories, f = files
    for r, d, f in os.walk(thisdir):

        # Go through each file    
        for file in f:

            # We're concerned with the results.json files only
            if file.startswith("results"):
                
                # Open up results file
                with open(os.path.join(r, file), "r") as f:

                    # Load in file
                    model_results = json.load(f)

                    # Set best accuracy to 0 at first
                    best_accuracy = 0

                    # If model's accuracy is better than best_accuracy, reset best accuracy and save model as best_model
                    if model_results['Accuracy'] > best_accuracy:
                        best_accuracy = model_results['Accuracy']
                        best_model = torch.load(os.path.join(r, "model.pt"))
                        torch.save(best_model, os.path.join('./results/','best_model.pt'))

                    #If model's accuracy is NOT better than best_accuracy, then keep best_accuracy as is and don't save model
                    else:
                        best_accuracy = best_accuracy
