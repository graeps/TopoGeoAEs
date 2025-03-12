import random
from collections import defaultdict
from typing import Tuple, Dict, Optional, Callable, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm.notebook import tqdm

class AutoEncoder(nn.Module):
    """
    Implementation of AutoEncoder (MLP).
    MLP might be more efficient than CNN for very shallow networks (just like in this case).
    """
    def __init__(self, latent_dim: int = 2):
        """
        Args:
            latent_dim: Latent dimension
        """
        super().__init__()

        # Expects input shape 1x28x28
        self._encoder = nn.Sequential(
            nn.Flatten(start_dim=-3),  # size: 28*28 = 784
            nn.Linear(784, 392),
            nn.ReLU(),
            nn.Linear(392, 196),
            nn.ReLU(),
            nn.Linear(196, latent_dim)
        )
        # encoder ~ posterior surrogate approximation

        self._decoder = nn.Sequential(
            nn.Linear(latent_dim, 196),
            nn.ReLU(),
            nn.Linear(196, 392),
            nn.ReLU(),
            nn.Linear(392, 784),
            nn.Unflatten(-1, (1, 28, 28))
        )
        # decoder - likelihood

    def show_encoder_transforms(self, x: torch.Tensor) -> None:
        """
        Prints list of encoder layers and output shapes. Useful for debugging and model validation.

        Args:
            x: Random input (with shape 1, 28, 28)
        """
        print('---Encoder transforms---')
        print(f'Input shape: {x.shape}')
        for i, layer in enumerate(self._encoder):
            layer_name = type(layer).__name__
            x = layer(x)
            print(f'[{i+1}] name={layer_name}, shape={x.shape}')

    def show_decoder_transforms(self, x: torch.Tensor) -> None:
        """
        Prints list of encoder layers and output shapes. Useful for debugging and model validation.

        Args:
            x: Random input (with shape 1, 28, 28)
        """
        print('---Decoder transforms---')
        print(f'[0] Input shape: {x.shape}')
        for i, layer in enumerate(self._decoder):
            layer_name = type(layer).__name__
            x = layer(x)
            print(f'[{i+1}] name={layer_name}, shape={x.shape}')
        print()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self._encoder(x)
        x_hat = self._decoder(z)
        return x_hat, z


ae = AutoEncoder()
ae.show_encoder_transforms(torch.randn(1, 28, 28, dtype=torch.float32))
ae.show_decoder_transforms(torch.randn(2, dtype=torch.float32))
