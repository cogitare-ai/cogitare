import torch


def seed(value):
    torch.manual_seed(value)
    torch.cuda.manual_seed(value)
