import torch
from torch.autograd import Variable
import logging
import coloredlogs
from itertools import repeat
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


def get_logger(name):
    _LOGGER = logging.getLogger(name)
    _LOGGER.addHandler(logging.NullHandler())
    coloredlogs.install(level='DEBUG', logger=_LOGGER)

    return _LOGGER


def tensorfy(*params, **kw):
    """Utility decorator to automatically convert a function parameters to tensor.

    You can pass the argument position, or the parameter name that you wish to be converted to tensor.

    Example::

        @tensorfy(0, 1, 2)
        def my_func(a, b, c, d, e):
            # will convert a, b, and c to tensor
            pass

        @tensorfy(0, 2, 'cmp')
        def my_func2(a, b, cmp=None):
            # will convert a and cmp (if defined) to tensor
            pass
    """
    def decorator(func):
        @functools.wraps(func)
        def f(*args, **kwargs):
            args = list(args)
            kwargs = dict(kwargs)

            kw.setdefault('dtype', torch.Tensor)
            for p in params:
                if isinstance(p, int):
                    if p < len(args):
                        args[p] = to_tensor(args[p], **kw)
                else:
                    if kwargs.get(p, None):
                        kwargs[p] = to_tensor(kwargs[p], **kw)
            return func(*args, **kwargs)

        return f
    return decorator


def assert_dim(tensor, name, expected):
    """Asserts if a tensor has a valid dimension.

    Args:
        tensor (torch.Tensor): tensor to verify
        name (str): verbose tensor name to diplay in the exception message.
        expected (list, tuple): list of ints with the accepted number of dimensions.

    Example::

        >>> a = torch.rand(2, 3)
        >>> b = torch.rand(2, 3, 4)
        >>> c = torch.rand(2, 3, 5, 5)

        >>> valid = (1, 2, 3)  # accept 1D, 2D, and 3D tensor
        >>> assert_dim(a, 'a', valid)
        >>> assert_dim(b, 'b', valid)
        >>> assert_dim(c, 'c', valid)
        ValueError: Expected 1D/2D/3D tensor on "c". Got a 4D tensor instead.
    """
    dim = tensor.dim()
    msg = 'Expected {} tensor on {}. Got a {}D tensor instead.'.format(
        '/'.join(str(d) + 'D' for d in expected),
        '"{}"'.format(name),
        dim)

    assert_raise(dim in expected, ValueError, msg)


def _as_2d(*items):
    result = []
    for (item, name) in items:
        assert_dim(item, name, (1, 2))

        if item.dim() == 1:
            item = item.view(1, -1)

        result.append(item)

    return result


def _assert_same_dim(*items):
    labels = ['"{}"'.format(v[1]) for v in items]
    items = [v[0] for v in items]

    size = items[0].size()
    valid = all(v.size() == size for v in items[1:])
    assert_raise(valid, ValueError, "The tensors {} should all have the same dimension.".format(
        '/'.join(labels)))


def number_parameters(model):
    """Counts the number of trainable and non-trainable parameters in the model.

    Args:
        model (Model): model with training parameters

    Returns:
        count (tuple): (trainable_params, non_trainable_params)
    """
    trainable = sum(np.prod(params.size()) for params in model.parameters() if params.requires_grad)
    non_trainable = sum(np.prod(params.size()) for params in model.parameters() if params.requires_grad is False)

    return trainable, non_trainable


def _training_mode(func, mode):
    @functools.wraps(func)
    def f(self, *args, **kwargs):
        assert isinstance(self, Module)
        default = self.training
        self.train(mode)
        value = func(self, *args, **kwargs)
        self.train(default)

        return value
    return f


def not_training(func):
    """Decorator to disable the training during execution. Must be used inside a Module class.

    Example::

        @not_training
        def my_func(self, arg1, arg2):
            # do something that cannot affect training, such as evaluating a
            # test set
            pass
    """
    return _training_mode(func, False)


def training(func):
    """Decorator do enable the training during execution. Must be used inside a Module class.

    Example::

        @training
        def my_func(self, arg1, arg2):
            # do something that can affect training, such as forward your train
            # data
            pass
    """
    return _training_mode(func, True)


def _get_first_item(data_list, depth=0):
    depth += 1
    if isinstance(data_list, (list, tuple)):
        if len(data_list) == 0:
            raise ValueError('Empty list')

        return _get_first_item(data_list[0], depth)
    return data_list, depth


def _list_to_tensor(data):
    item, depth = _get_first_item(data)
    converter = {
        int: torch.LongTensor,
        float: torch.DoubleTensor,
        np.ndarray: lambda x: torch.from_numpy(np.stack(x))
    }

    if torch.is_tensor(item):
        if depth != 2:
            raise ValueError('Cannot convert nested list of tensors')
        tensor = torch.stack(data)
    elif type(item) in converter:
        tensor = converter[type(item)](data)
    else:
        raise ValueError('Invalid data type: {}'.format(type(item).__name__))

    return tensor


def _convert_tensor(tensor, dtype, use_cuda):
    if dtype:
        tensor = tensor.type(dtype)

    if get_cuda(use_cuda):
        tensor = tensor.cuda()

    return tensor


def to_tensor(data, dtype=None, use_cuda=None):
    """Try to conver a variable to a torch tensor.

    Data can be:

        - list of Tensors
        - list of list of list (...) of int/float
        - list of numpy arrays
        - numpy array
        - tensor

    And this will return a torch tensor.

    Args:
        data (list, numpy.ndarray, torch.Tensor): the data to be converted.
        dtype (torch.Tensor): if provided, the output will have the same type
            as this class.
        use_cuda (bool): if True, the tensor ``.cuda()`` will be returned. If None, the
            default value setted by :func:`~cogitare.utils.set_cuda` will be used.

    Returns:
        tensor (torch.Tensor): the data converted to tensor type.

    Examples::

        >>> a = [1, 2, 3]
        >>> b = [[1, 2, 3], [4, 5, 6]]
        >>> to_tensor(a)
         1
         2
         3
        [torch.LongTensor of size 3]
        >>> to_tensor(b, torch.FloatTensor)
         1  2  3
         4  5  6
        [torch.FloatTensor of size 2x3]

        >>> c = [torch.Tensor([1, 2, 3]), torch.Tensor([4, 5, 6])]
        >>> to_tensor(c)
         1  2  3
         4  5  6
        [torch.FloatTensor of size 2x3]

        >>> d = [np.asarray([1, 2, 3]), np.asarray([4, 5, 6])]
        >>> to_tensor(d)
         1  2  3
         4  5  6
        [torch.LongTensor of size 2x3]
    """
    # if list, cast it to the compatible tensor type
    converter = {
        list: _list_to_tensor,
        tuple: _list_to_tensor,
        np.ndarray: torch.from_numpy,
        Variable: lambda x: x
    }

    if type(data) in converter:
        tensor = converter[type(data)](data)
    elif torch.is_tensor(data):
        tensor = data
    else:
        raise ValueError('Invalid data type: {}'.format(type(data).__name__))

    tensor = _convert_tensor(tensor, dtype, use_cuda)

    return tensor


def to_variable(data, dtype=None, use_cuda=None, **kwargs):
    """Similar to :func:`~cogitare.utils.to_tensor`, this function converts
    the ``data`` o a :class:`torch.autograd.Variable`.

    If first convert the data to tensor using :func:`to_tensor`, and then cast it to a
    variable.

    Args:
        data: the data to convert.
        dtype: a tensor class to use as the type of the data
        use_cuda (bool): if True, move the tensor to cuda.
        kwargs: extra parameters to the Variable class.

    Example:

        >>> a = [[1, 2,3], [4, 5, 6]]
        >>> to_variable(a)
        Variable containing:
         1  2  3
         4  5  6
        [torch.LongTensor of size 2x3]
    """
    tensor = to_tensor(data, dtype=dtype, use_cuda=use_cuda)
    if not isinstance(tensor, Variable):
        tensor = Variable(tensor, **kwargs)
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
    it's necesary to automatically convert the Tensor/Module to cuda.
    """
    global _CUDA_ENABLED
    _CUDA_ENABLED = cuda


def _ntuple(item, n):
    if isinstance(item, (list, tuple)):
        return item
    return tuple(repeat(item, n))
