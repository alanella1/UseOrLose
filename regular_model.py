import torch
import torch.nn as nn

INPUT_DIM = 561
OUTPUT_DIM = 6


class SimpleFeedForward(nn.Module):
    def __init__(self, hidden_dim=256):
        super(SimpleFeedForward, self).__init__()

        self.linear1 = nn.Linear(INPUT_DIM, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.drop1 = nn.Dropout(0.2)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.drop2 = nn.Dropout(0.2)
        self.linear3 = nn.Linear(hidden_dim, OUTPUT_DIM)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.drop1(self.relu(self.bn1(self.linear1(x))))
        x = self.drop2(self.relu(self.bn2(self.linear2(x))))
        x = self.linear3(x)  # No softmax, handled in loss
        return x
