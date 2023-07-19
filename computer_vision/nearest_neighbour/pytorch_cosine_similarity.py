"""Finding exact nearest vector.
"""
import torch
from torch import nn


# device = "cpu"
device = "cuda"

input1 = torch.tensor([[0.1, 0.2, 0.5, 0.8]], device=device)

input2 = torch.tensor([
    [0.5, 0.8, 0.8, 0.2],
    [0.2, 0.3, 0.6, 0.7],
    [0.2, 0.3, 0.4, 0.9],
    [0.12, 0.3, 0.5, 0.8],
    [0.1, 0.2, 0.51, 0.81],
    [0.9, 0.9, 0.1, 0.1],
], device=device)


cos = nn.CosineSimilarity(dim=1, eps=1e-6).to(device)
output = cos(input1, input2)
print(output)
