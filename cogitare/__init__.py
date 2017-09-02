import torch
from cogitare.core import PluginInterface
from cogitare.core import Model


__all__ = ['Model', 'PluginInterface']


def seed(value):
    torch.manual_seed(value)
    torch.cuda.manual_seed(value)
