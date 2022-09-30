# transforms.py

# torch transforms

from torchvision import transforms

def transform_torch():
    ts = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    return(ts)