#src/regressor_model.py
import torch.nn as nn
import torch.nn.functional as F


class RegressionModel(nn.Module):
    """
    Simple feed-forward regression network:
      - FC(embedding_dim -> 128) + ReLU
      - FC(128 -> 64) + ReLU
      - FC(64 -> 1) (linear output)
    """
    def __init__(self, input_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
