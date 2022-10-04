from ser.data import test_dataloader
from ser.transforms import flip
from ser.select_transforms import _select_test_image

#create example dataset
dataset = test_dataloader(10, flip)



