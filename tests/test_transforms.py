from ser.transforms import flip
import numpy as np
from torch import tensor, equal

def test_flip():

    data = tensor(np.array([[0,0,1],[0,0,1],[0,0,1]]))
    assert equal(flip()(data),tensor(np.array([[1,0,0],[1,0,0],[1,0,0]])))

