from ser.data import test_dataloader
from ser.transforms import transforms, normalize, flip

def _select_test_image(label, flip_image, normalize_image):
    # TODO `ts` is a list of transformations that will be applied to the loaded
    # image. This works... but in order to add a transformation, or change one,
    # we now have to come and edit the code... which sucks. What if we could
    # configure the transformations via the cli?

    if flip_image and normalize_image:
        ts = [flip, normalize]
    elif flip_image:
        ts = [flip]
    elif normalize_image:
        ts = [normalize]
    else:
        ts = []

    dataloader = test_dataloader(1, transforms(*ts))
    images, labels = next(iter(dataloader))
    while labels[0].item() != label:
        images, labels = next(iter(dataloader))
    return images