import torch
from torch.distributions import Distribution
from torch.distributions import constraints


# TODO: add matrix parameter and define Uniform distr on parallelogram
class CubeUniform(Distribution):
    def __init__(self, low, high):
        # low and high should be tensors of shape [d] defining the bounds of the cube
        self.low = low
        self.high = high
        self.d = low.shape[0]  # Dimension
        super().__init__()

    arg_constraints = {
        'low': constraints.real,
        'high': constraints.real,
        "d": constraints.positive,
    }

    def sample(self, sample_shape=torch.Size()):
        # Sample uniformly from [low, high]^d
        return self.low + (self.high - self.low) * torch.rand(sample_shape + (self.d,))

    def log_prob(self, value):
        # Check that value is within bounds
        if torch.any(value < self.low) or torch.any(value > self.high):
            return torch.full_like(value, float('-inf'))  # Log probability is -inf outside the bounds

        # Uniform distribution has constant density within the cube
        volume = torch.prod(self.high - self.low)
        return -torch.log(volume).expand(value.shape[0])  # Uniform density is 1/volume

    def entropy(self):
        # Entropy for a uniform distribution on a cube: H = d * log(b - a)
        volume = torch.prod(self.high - self.low)
        return torch.log(volume).expand(1)  # d-dimensional cube entropy
