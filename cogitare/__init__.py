import torch
import random
import numpy as np
from cogitare.core import PluginInterface
from cogitare.core import Model


__all__ = ['Model', 'PluginInterface']


def seed(value):
    torch.manual_seed(value)
    torch.cuda.manual_seed(value)
    random.seed(value)
    np.random.seed(value)
