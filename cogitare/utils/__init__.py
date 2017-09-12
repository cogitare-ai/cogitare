import torch
import functools
import numpy as np
from torch.nn import Module


_CUDA_ENABLED = False


class StopTraining(Exception):
    """This is an Exception to stop the training process of a :class:`cogitare.Model`.

    While running the :meth:`cogitare.Model.learn`, if a plugin raises this exception,
    the model will run the plugins in the ``on_stop_training`` hook, and stop the training.
    """
    pass


def not_training(func):
    """Decorator to disable the training during execution. Must be used inside a Module class.

    Example::

        @not_training
        def my_func(arg1, arg2):
            # do something that cannot affect training, such as evaluating a
            # test set
            pass
    """
    @functools.wraps(func)
    def f(self, *args, **kwargs):
        assert isinstance(self, Module)
        default = self.training
        self.train(False)
        value = func(self, *args, **kwargs)
        self.train(default)

        return value

    return f


def training(func):
    """Decorator to enable the training during execution. Must be used inside a Module class.

    Example::

        @training
        def my_func(arg1, arg2):
            # do something that can affect training, such as forward your train
            # data
            pass
    """
    @functools.wraps(func)
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

    def get_first_item(data_list, depth=0):
        depth += 1
        if isinstance(data_list, list):
            if len(data_list) == 0:
                raise ValueError('Empty list')

            return get_first_item(data_list[0], depth)
        return data_list, depth

    if isinstance(data, list):
        item, depth = get_first_item(data)

        if torch.is_tensor(item):
            if depth != 2:
                raise ValueError('Cannot convert nested list of tensors')
            for l in data:
                l.unsqueeze_(0)
            tensor = torch.cat(data)
        elif isinstance(item, int):
            tensor = torch.LongTensor(data)
        elif isinstance(item, float):
            tensor = torch.DoubleTensor(data)
        elif isinstance(item, np.ndarray):
            tensor = torch.from_numpy(np.stack(data))
        else:
            raise ValueError('Invalid data type: {}'.format(type(item).__name__))
    elif isinstance(data, np.ndarray):
        tensor = torch.from_numpy(data)
    elif torch.is_tensor(data):
        tensor = data
    else:
        raise ValueError('Invalid data type: {}'.format(type(data).__name__))

    if tensor_klass:
        tensor = tensor.type_as(tensor_klass())

    if get_cuda(use_cuda):
        tensor = tensor.cuda()

    return tensor


def assert_raise(valid, exception, msg):
    """Shortcut to assert if something is valid. If invalid,
    raises the exception with the provided message.

    Example::

        >>> assert_raise(1 == 2, ValueError, 'The values must be equal')
        ValueError: The values must be equal
    """
    if not valid:
        raise exception(msg)


def get_cuda(cuda=None):
    """Get the default cuda enabled/disabled variable.

    Args:
        cuda (bool): if a boolean is provided, its value will be returned.

    Returns:
        enabled (bool): the ``cuda`` parameter, if provided, or the enaled/disabled status.
    """
    if cuda is None:
        return _CUDA_ENABLED
    return cuda


def set_cuda(cuda):
    """Set Cogitare default cuda enabled/disabled.

    This value is used by Cogitare's models and some functions to determine if
    it's necessary to convert the Tensor/Module to cuda automatically.
    """
    global _CUDA_ENABLED
    _CUDA_ENABLED = cuda
