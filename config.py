import os
import numpy as np
import pandas as pd
import torch
from torch import nn, optim


config = {
  "num_factors": 16,
  "hidden_layers": [64, 32, 16],
  "embedding_dropout": 0.05,
  "dropouts": [0.3, 0.3, 0.3],
  "learning_rate": 1e-3,
  "weight_decay": 1e-5,
  "batch_size": 8,
  "num_epochs": 3,
  "total_patience": 30,
  "data_path": "../data/",
  "model_path": "../model/"
}
