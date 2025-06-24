import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

import torch

# Check if MPS (Apple GPU) is available
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# Creating tensors https://docs.pytorch.org/docs/stable/tensors.html

# scalar
scalar = torch.tensor(7)

# vector
vector = torch.tensor([7, 2])

# MATRIX
MATRIX = torch.tensor([[1, 2], [5, 8]])

# TENSOR
TENSOR =torch.tensor([[[1, 2], [5, 9]], [[3, 1], [6, 9]]], device=device)

# Random TENSOR of a given shape
R_TENSOR = torch.rand(3, 4, 2)


# Move tensor to GPU
RANGE_TENSOR = torch.arange(5, 101, 5, device=device)

print(TENSOR.ndim)
print(TENSOR)
print(TENSOR.shape)
print(TENSOR[1])
print(TENSOR.device)