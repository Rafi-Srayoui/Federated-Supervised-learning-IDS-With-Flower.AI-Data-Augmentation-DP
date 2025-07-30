# model.py
import torch
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix


class Net(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), 
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):                     # x : (N, input_dim)
        return self.net(x)                    # logits


# Keep these in model.py or a tiny util.py
import numpy as np

def get_parameters(model):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

def set_parameters(model, params):
    state_dict = {
        k: torch.tensor(v, dtype=param.dtype)
        for (k, param), v in zip(model.state_dict().items(), params)
    }
    model.load_state_dict(state_dict, strict=True)
