import torch
from torch.nn import Module


_CUDA_ENABLED = False


class StopTraining(Exception):
    pass


def not_training(func):
    """decorator do disable the training during execution. Must be used inside a Module class
    """
    def f(self, *args, **kwargs):
        assert isinstance(self, Module)
        default = self.training
        self.train(False)
        value = func(self, *args, **kwargs)
        self.train(default)

        return value

    return f


def training(func):
    """decorator do enable the training during execution. Must be used inside a Module class
    """
    def f(self, *args, **kwargs):
        assert isinstance(self, Module)
        default = self.training
        self.train(True)
        value = func(self, *args, **kwargs)
        self.train(default)

        return value

    return f


def to_tensor(data, tensor_klass=None, use_cuda=None):
    # if list, cast it to the compatible tensor type
    tensor = None

    if isinstance(data, list):
        item = data[0]

        if torch.is_tensor(item):
            for l in data:
                l.unsqueeze_(0)
            tensor = torch.cat(data)
        elif isinstance(item, int):
            tensor = torch.LongTensor(data)
        elif isinstance(item, float):
            tensor = torch.DoubleTensor(data)
        else:
            raise ValueError('Invalid data type: {}'.format(type(item)))
    elif torch.is_tensor(data):
        tensor = data
    else:
        raise ValueError('Invalid data type: {}'.format(type(data)))

    if tensor_klass:
        tensor = tensor.type_as(tensor_klass())

    if get_cuda(use_cuda):
        tensor = tensor.cuda()

    return tensor


def assert_raise(valid, exception, msg):
    """if boolean is False, raises a excetion using the provided message
    """
    if not valid:
        raise exception(msg)


def get_cuda(cuda=None):
    """get Cogitare default cuda enabled/disabled."""
    if cuda is None:
        return _CUDA_ENABLED
    return cuda


def set_cuda(cuda):
    """set Cogitare default cuda enabled/disabled."""
    global _CUDA_ENABLED
    _CUDA_ENABLED = cuda
