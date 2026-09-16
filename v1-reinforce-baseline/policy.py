import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        # state = (x, y, battery, survivors_fraction)
        self.fc = nn.Linear(4, 5)

    def forward(self, x):
        return F.softmax(self.fc(x), dim=-1)
