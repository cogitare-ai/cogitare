from cogitare.data.dataholder import AbsDataHolder
from six import add_metaclass
from six.moves import zip_longest
from abc import ABCMeta
import torch
import numpy


@add_metaclass(ABCMeta)
class SequentialAbsDataHolder(AbsDataHolder):
    """
    This class is an extension of :class:`~cogitare.data.AbsDataHolder` to support
    sequential data, iterating over the batches and over timesteps. Its the recommended
    interface for using with :class:`~cogitare.SequentialModel`.

    An abstract object that acts as a data holder. A data holder is a utility to hold
    datasets, which provide some simple functions to work with the dataset, such as
    sorting, splitting, dividing it into chunks, loading batches using multi-thread, and so on.

    It's the recommended way to pass data to Cogitare's models because it already
    provides a compatible interface to iterate over batches and timesteps.

    To improve the performance, the data holder loads batches using multiprocessing and multithreading
    data loader with `Dask <http://dask.pydata.org/>`_.

    Usually, this object should not be used directly, only if you are developing a custom
    data loader. Cogitare already provides the following implementations for the most
    common data types:

        - Sequence from Tensors: :class:`~cogitare.data.SequentialTensorHolder`
        - Sequence from Numpy: :class:`~cogitare.data.SequentialNumpyHolder`
        - Sequence from Callable (functions that receive the sample id, and returns its
          data): :class:`~cogitare.data.SequentialCallableHolder`
        - :class:`~cogitare.data.SequentialAutoHolder`: inspect the data to choose one of the available data holders.

    Args:
        data (torch.Tensor, numpy.ndarray, callable): the data to be managed by the data holder.
        batch_size (int): the size of the batch.
        shuffle (bool): if True, shuffles the dataset after each iteration.
        drop_last (bool): if True, then skip the batch if its size is lower that **batch_size** (can
            occur in the last batch).
        total_samples (int): the number of total samples. If provided, this will limit the
            number of samples to be accessed in the data.
        mode (str): must be one of: 'sequential', 'threaded', 'multiprocessing'. Use one of them
            to choose the batch loading methods. Take a loook
            here: https://dask.pydata.org/en/latest/scheduler-choice.html for an overview
            of the advantage of each mode.
        padding_value: this value will be used to pad sequences with different
            sizes in the same batch. When loading a batch, all sequences will have
            the same size. The padding_value is added to the right of each sequence to match the size
            of the longest sequence in the batch.
        sort_by_len (int): if True, the sequences in the batch will be sorted by decreasing size. This is
            useful to be load data for torch rnn.PackedSequence. If True, the iterator will return a
            tuple with (data, original indices, sizes).
    """

    @property
    def padding_value(self):
        """The value used to pad sequences with different size in the same batch
        """
        return self._padding_value

    @padding_value.setter
    def padding_value(self, value):
        self._padding_value = value

    def __init__(self, *args, **kwargs):
        self._padding_value = kwargs.pop('padding_value', None)
        self._sort_by_len = kwargs.pop('sort_by_len', False)

        super(SequentialAbsDataHolder, self).__init__(*args, **kwargs)

    def __iter__(self):
        return self

    def __next__(self):
        batch = super(SequentialAbsDataHolder, self).__next__()
        if self._sort_by_len:
            lengths = [len(v) for v in batch]
            indices = [v[0] for v in sorted(enumerate(lengths), reverse=True, key=lambda x: x[1])]

            sorted_batch = [batch[i] for i in indices]
            sorted_lengths = [lengths[i] for i in indices]

            data = zip_longest(*sorted_batch, fillvalue=self.padding_value)
            return list(data), indices, sorted_lengths
        else:
            data = zip_longest(*batch, fillvalue=self.padding_value)
            return list(data)

    next = __next__


class SequentialCallableHolder(SequentialAbsDataHolder):
    """SequentialCallableHolder is a data holder for abritary data type.

    As data input, it uses a callable that receive the sample index as parameter,
    and must return the sample (an interator with all the timesteps for the sample).

    It can be used to load non-Tensor or non-numpy datasets, such as texts, dicts, and anything else.
    You are free to use SequentialCallableHolder with any data type. As a requirement for sequential models,
    the returned value must be an iterator containing the timesteps, and have the length attribute.

        .. note:: When using SequentialCallableHolder, you must specify the number of samples
            in the dataset. The callable will be called asking for samples from 0 to (total_samples - 1).

    Example::

        >>> def load_sample(idx):
        ...     # the idx'th sample with 3 features per timestep, and #idx timesteps
        ...     return [(i, i, i) for i in range(idx)]

        >>> # when using the SequentialCallableHolder. you must pass the number of samples to
        >>> # be loaded.

        >>> # you can set the total_samples using the parameter in the constructor
        >>> data = SequentialCallableHolder(load_sample, batch_size=2, total_samples=5)
        >>> # or by setting the property
        >>> data.total_samples = 5

        >>> batch = next(data)
        >>> batch
        [((0, 0, 0), (0, 0, 0)), (None, (1, 1, 1)), (None, (2, 2, 2)), (None, (3, 3, 3))]
        >>> for timestep, data in enumerate(batch, 1):
        ...   print('Current timestep: ' + str(timestep))
        ...   print(data)
        Current timestep: 1
        ((0, 0, 0), (0, 0, 0))
        Current timestep: 2
        (None, (1, 1, 1))
        Current timestep: 3
        (None, (2, 2, 2))
        Current timestep: 4
        (None, (3, 3, 3))

        >>> # in the example above, the first sequence had length 1, and the second
        >>> # length 4. The first one was padded with None.

        >>> # to pad with a different value, use:
        >>> data = SequentialCallableHolder(load_sample, batch_size=2,
        ...                                 total_samples=5, padding_value=-1)
        >>> batch = next(data)
        >>> batch
        [((0, 0, 0), (0, 0, 0)), ((1, 1, 1), (1, 1, 1)), ((2, 2, 2), -1), ((3, 3, 3), -1)]
        >>> for timestep, data in enumerate(batch, 1):
        ...   print('Current timestep: ' + str(timestep))
        ...   print(data)
        Current timestep: 1
        ((0, 0, 0), (0, 0, 0))
        Current timestep: 2
        ((1, 1, 1), (1, 1, 1))
        Current timestep: 3
        ((2, 2, 2), -1)
        Current timestep: 4
        ((3, 3, 3), -1)
    """

    def get_sample(self, key):
        return self._data(key)


class SequentialTensorHolder(SequentialAbsDataHolder):
    """A data holder for sequences in :class:`torch.Tensor`.

    The tensor must have the shape (N, S, \*), where:

        - N is batch size (number of samples in the tensor);
        - S is the sequence length

    The :class:`~cogitare.data.SequentialTensorHolder` will first iterate over the batches,
    getting ``batch_size`` samples from the first dimension. And the will iterate over the second
    dimension of the mini-batch.

    Example::

        >>> tensor = torch.Tensor([[1,2,3], [4,5,6], [7,8,9]])
        >>> tensor
         1  2  3
         4  5  6
         7  8  9
        [torch.FloatTensor of size 3x3]
        >>> data = SequentialTensorHolder(tensor, batch_size=2)

        >>> batch = next(data)
        >>> batch
        [(1.0, 7.0), (2.0, 8.0), (3.0, 9.0)]

        >>> for timestep, data in enumerate(batch, 1):
        ...   print('Current timestep: ' + str(timestep))
        ...   print(data)
        Current timestep: 1
        (1.0, 7.0)
        Current timestep: 2
        (2.0, 8.0)
        Current timestep: 3
        (3.0, 9.0)
    """

    def __init__(self, *args, **kwargs):
        super(SequentialTensorHolder, self).__init__(*args, **kwargs)
        self._total_samples = len(self._data)

    def get_sample(self, key):
        return self._data[key]


def SequentialNumpyHolder(data, *args, **kwargs):
    """
    When creating the object, it converts the numpy data to Tensor using
    :func:`torch.from_numpy` and then creates an :class:`~cogitare.data.SequentialTensorHolder`
    instance.
    """
    data = torch.from_numpy(data)

    return SequentialTensorHolder(data=data, *args, **kwargs)


def SequentialAutoHolder(data, *args, **kwargs):
    """Check the data type to infer which sequential data holder to use.
    """
    if isinstance(data, numpy.ndarray):
        return SequentialNumpyHolder(data, *args, **kwargs)
    if torch.is_tensor(data):
        return SequentialTensorHolder(data, *args, **kwargs)
    if callable(data):
        return SequentialCallableHolder(data, *args, **kwargs)
    raise ValueError('Unable to infer data type!')
