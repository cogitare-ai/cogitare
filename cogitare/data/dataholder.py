import torch
import math
from abc import ABCMeta, abstractmethod
from cogitare import utils
from six import add_metaclass
import numpy
from dask import threaded, delayed, compute, multiprocessing


@add_metaclass(ABCMeta)
class AbsDataHolder(object):
    """
    An abstract object that acts as a data holder. A data holder is a utility to hold
    datasets, which provide some simple functions to work with the dataset, such as
    sorting, splitting, dividing it into chunks, loading batches using multi-thread, and so on.

    It's the recommended way to pass data to Cogitare's models because it already
    provides a compatible interface to iterate over batches.

    To improve the performance, the data holder loads batches using a multi-threaded
    loader with `Dask <http://dask.pydata.org/>`_.

    Usually, this object should not be used directly, only if you are developing a custom
    data loader. Cogitare already provides the following implementations for the most
    common data types:

        - Tensors: :class:`~cogitare.data.TensorHolder`
        - Numpy: :class:`~cogitare.data.NumpyHolder`
        - Callable (functions that receive the sample id, and returns its
          data): :class:`~cogitare.data.CallableHolder`
        - :class:`~cogitare.data.AutoHolder`: inspect the data to choose one of the available data holders.

    Args:
        data (torch.Tensor, numpy.ndarray, callable): the data to be managed by the data holder.
        batch_size (int): the size of the batch.
        shuffle (bool): if True, shuffles the dataset after each iteration.
        drop_last (bool): if True, then skip the batch if its size is lower that **batch_size** (can
            occur in the last batch).
    """

    @property
    def total_samples(self):
        """Returns the number of individual samples in this dataset.
        """
        return self._total_samples

    @total_samples.setter
    def total_samples(self, value):
        if hasattr(self._data, '__len__'):
            size = len(self._data)
        else:
            size = None

        if size is not None:
            utils.assert_raise(value <= size, ValueError,
                               'The value must be lesser or equal to the'
                               'length of the input data')
        self._total_samples = size

    @property
    def indices(self):
        if self._indices is None:
            self._indices = numpy.arange(self.total_samples)

        return self._indices

    def __init__(self, data=None, batch_size=1, shuffle=True, drop_last=False,
                 total_samples=None, mode='sequential'):
        valid_modes = ['threaded', 'multiprocessing', 'sequential']
        utils.assert_raise(data is not None, ValueError, 'data cannot be None')
        utils.assert_raise(mode in valid_modes, ValueError,
                           '"mode" must be one of: ' + ', '.join(valid_modes))

        self._indices = None
        self._total_samples = total_samples
        self._remaining_samples = None

        self._data = data
        self._batch_size = batch_size

        self._current_batch = 0
        self._drop_last = drop_last
        self._shuffle = shuffle

        self._requires_reset = True

        if mode == 'sequential':
            self._get = None
        elif mode == 'threaded':
            self._get = threaded.get
        else:
            self._get = multiprocessing.get

    def __repr__(self):
        """Using repr(data) or str(data), display the shape of the data.
        """

        return '{} with {}x{} samples'.format(type(self).__name__, len(self), self._batch_size)

    def __getitem__(self, key):
        """Get a sample in the dataset using its indices.

        Example::

            sample = data[0]
            sample2 = data[1]
        """
        return self.get_sample(self.indices[key])

    def _get_batch(self):
        if self._requires_reset:
            self.reset()

        batch_size = min(self._batch_size, self._remaining_samples)

        if batch_size < self._batch_size and self._drop_last:
            self._requires_reset = True
            raise StopIteration

        if batch_size == 0:
            self._requires_reset = True
            raise StopIteration

        if self._get:
            # use dask
            jobs = (delayed(self.__getitem__, traverse=False)
                    (self._current_batch * self._batch_size + i) for i in range(batch_size))
            results = compute(jobs, get=self._get)[0]
        else:
            results = [self.__getitem__(self._current_batch * self._batch_size + i)
                       for i in range(batch_size)]

        self._current_batch += 1
        self._remaining_samples -= batch_size

        return results

    @abstractmethod
    def get_sample(self, key):
        pass

    def __len__(self):
        """Return the number of batches in the dataset.
        """
        if self._drop_last:
            return self.total_samples // self._batch_size
        else:
            return (self.total_samples + self._batch_size - 1) // self._batch_size

    def __iter__(self):
        """Creates an iterator to iterate over batches in the dataset.

        After each iteration over the batches, the dataset will be shuffled if
        the **shuffle** parameter is True.

        Example::

            for sample in data:
                print(sample)
        """
        return self

    def __next__(self):
        return self._get_batch()

    next = __next__

    def reset(self):
        """Reset the batch iterator.

        This method returns the iterator to the first sample, and shuffle the
        dataset if shuffle is enabled.
        """
        self._requires_reset = False
        self._current_batch = 0
        self._remaining_samples = self.total_samples
        if self._shuffle:
            self.shuffle()

    def shuffle(self):
        """Shuffle the samples in the dataset.

        This operation will not affect the original data.
        """
        numpy.random.shuffle(self.indices)

    def split(self, ratio):
        """Split the data holder into two data holders.

        The first one will receive *total_samples * ratio* samples, and the second
        data holder will receive the remaining samples.

        Args:
            ratio (:obj:`float`): ratio of the split. Must be between 0 and 1.

        Returns:
            (data1, data2): two data holder, in the same type that the original.

        Example::

            >>> print(data)
            TensorHolder with 875x64 samples
            >>> data1, data2 = data.split(0.8)
            >>> print(data1)
            TensorHolder with 700x64 samples
            >>> print(data2)
            TensorHolder with 175x64 samples
        """
        utils.assert_raise(0 < ratio < 1, ValueError, '"ratio" must be between 0 and 1')

        pos = math.floor(self.total_samples * ratio)

        data1 = type(self)(data=self._data, batch_size=self._batch_size,
                           shuffle=self._shuffle, drop_last=self._drop_last)

        data2 = type(self)(data=self._data, batch_size=self._batch_size,
                           shuffle=self._shuffle, drop_last=self._drop_last)

        data1._indices = self.indices[:pos]
        data2._indices = self.indices[pos:]
        data1._total_samples = pos
        data2._total_samples = self.total_samples - pos

        return data1, data2

    def split_chunks(self, n):
        """Split the data holder into N data holders with the sample number of samples each.

        Args:
            n (int): number of new splits.

        Returns:
            output (list): list of N data holders.

        Example::

            >>> print(data)
            TensorHolder with 875x64 samples
            >>> data1, data2, data3 = data.split_chunks(3)
            >>> print(data1)
            TensorHolder with 292x64 samples
            >>> print(data2)
            TensorHolder with 292x64 samples
            >>> print(data3)
            TensorHolder with 292x64 samples
        """
        size = self.total_samples // n

        data = []

        for i in range(n):
            begin, end = i * size, min((i + 1) * size, self.total_samples)

            holder = type(self)(data=self._data, batch_size=self._batch_size,
                                shuffle=self._shuffle, drop_last=self._drop_last)
            holder._indices = self.indices[begin:end]
            holder._total_samples = end - begin

            data.append(holder)

        return data


class CallableHolder(AbsDataHolder):
    """CallableHolder is a data holder for abritary data type.

    As data input, it uses a callable that receive the sample index as parameter,
    and must return the sample.

    It can be used to load non-Tensor or non-numpy datasets, such as texts, dicts, and anything else.
    You are free to use CallableHolder with any data type.

        .. note:: When using CallableHolder, you must specify the number of samples
            in the dataset. The callable will be called asking for samples from 0 to (total_samples - 1).

    Example::

        >>> def load_sample(idx):
        ...     return list(range(idx, idx + 10))

        >>> # when using the CallableHolder. you must pass the number of samples to
        >>> # be loaded.

        >>> # you can set the total_samples using the parameter in the constructor
        >>> data = CallableHolder(load_sample, batch_size=8, total_samples=20)
        >>> # or by setting the property
        >>> data.total_samples = 20

        >>> next(data)
        [[8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
         [9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
         [6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
         [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
         [13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
         [7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
         [18, 19, 20, 21, 22, 23, 24, 25, 26, 27],
         [17, 18, 19, 20, 21, 22, 23, 24, 25, 26]]

    """

    @property
    def total_samples(self):
        """The number of samples in the dataset. You must set this value before accessing the data.
        """
        return self._total_samples

    @total_samples.setter
    def total_samples(self, total_samples):
        utils.assert_raise(total_samples >= 1, ValueError,
                           'number of samples must be greater or equal to 1')
        self._total_samples = total_samples
        self._remaining_samples = total_samples

    def __init__(self, *args, **kwargs):
        total_samples = kwargs.pop('total_samples', None)
        self._total_samples = total_samples
        super(CallableHolder, self).__init__(*args, **kwargs)

    def get_sample(self, key):
        return self._data(key)


class TensorHolder(AbsDataHolder):
    """
    A data holder to work with :class:`torch.Tensor` objects.

    Example::

        >>> tensor = torch.Tensor([[1,2,3], [4,5,6], [7,8,9]])
        >>> tensor
        1  2  3
        4  5  6
        7  8  9
        [torch.FloatTensor of size 3x3]
        >>> data = TensorHolder(tensor, batch_size=2)
        >>> for sample in data:
        ...     print('Sample:')
        ...     print(sample)
        ...     print('Sample as tensor:')
        ...     print(utils.to_tensor(sample))
        Sample:
        [
         7
         8
         9
        [torch.FloatTensor of size 3]
        ,
         4
         5
         6
        [torch.FloatTensor of size 3]
        ]
        Sample as tensor:

         7  8  9
         4  5  6
        [torch.FloatTensor of size 2x3]

        Sample:
        [
         1
         2
         3
        [torch.FloatTensor of size 3]
        ]
        Sample as tensor:

         1  2  3
        [torch.FloatTensor of size 1x3]
    """

    def __init__(self, *args, **kwargs):
        super(TensorHolder, self).__init__(*args, **kwargs)
        size = len(self._data)

        if self._total_samples is None:
            self._total_samples = size
        else:
            utils.assert_raise(self.total_samples <= size, ValueError,
                               'The total_samples must be lesser or equal to the'
                               'length of the input data')

    def get_sample(self, key):
        return self._data[key]


def NumpyHolder(data, *args, **kwargs):
    """
    When creating the object, it converts the numpy data to Tensor using
    :func:`torch.from_numpy` and then creates an :class:`~cogitare.data.TensorHolder`
    instance.
    """
    data = torch.from_numpy(data)

    return TensorHolder(data=data, *args, **kwargs)


def AutoHolder(data, *args, **kwargs):
    """Check the data type to infer which data holder to use.
    """
    if torch.is_tensor(data):
        return TensorHolder(data, *args, **kwargs)
    elif isinstance(data, numpy.ndarray):
        return NumpyHolder(data, *args, **kwargs)
    elif callable(data):
        return CallableHolder(data, *args, **kwargs)
    else:
        raise ValueError('Unable to infer data type!')
