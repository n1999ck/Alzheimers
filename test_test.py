import sys

import torch.nn as nn
import torch.utils
import torch.utils.data as data_utils
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.utils.data import random_split
#https://stackoverflow.com/questions/55691819/why-does-dim-1-return-row-indices-in-torch-argmax

assert sys.version_info >= (3, 5) # Version of python must be above 3.5

#Science Kit Learn >= 0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"

# Common imports
import numpy as np
import os
import pandas as pd

x = torch.tensor([[-0.9175,  0.8993]])
_, predictedx = torch.max(x.data, 1)

y = torch.tensor([[-0.9175,  0.8993]])
_, predictedy = torch.max(x.data, 0)

print(predictedx)
print(predictedy)


arr = [[1,2],[3,4],[5,6],[7,8],[9,10],[11,12],[13,14],[15,16]]
arr = torch.tensor(arr)
print(arr.shape)

arr0 = arr.squeeze(0)
arr1 = arr.squeeze(1)

print(arr0)
print(arr1)
print(arr0.shape)
print(arr1.shape)