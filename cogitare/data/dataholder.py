import torch
import random
import math
from abc import ABCMeta, abstractmethod
from cogitare import utils
from six import add_metaclass
import numpy
from dask import threaded, delayed, compute


@add_metaclass(ABCMeta)
class AbsDataHolder(object):
    """
    Abstract object that acts as a data holder. A data holder is a utility to hold
    datasets, that provide some simple functions to work with the dataset, such as
    sorting, spliting, dividing it into chunks, loading batches using multi-thread, and so on.

    It's the recommended way to pass data to Cogitare's models, because it already
    provides an compatible interface to iterate over batches.

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
        drop_last (bool): if True, skip the batch if its size is lower then **batch_size** (can
            occours in the last batch).
    """

    @property
    def total_samples(self):
        """Returns the number of individual samples in this dataset.
        """
        return self._total_samples

    @property
    def indices(self):
        if not self._indices:
            self._indices = list(range(self.total_samples))

        return self._indices

    def __init__(self, data=None, batch_size=1, shuffle=True, drop_last=False):
        utils.assert_raise(data is not None, ValueError, 'data cannot be None')

        self._indices = None
        self._total_samples = None
        self._remaining_samples = None

        self._data = data
        self._batch_size = batch_size

        self._current_batch = 0
        self._drop_last = drop_last
        self._shuffle = shuffle

        self._requires_reset = True

    def __repr__(self):
        """Using repr(data) or str(data), display the shape of the data.
        """

        return '{} with {}x{} samples'.format(type(self).__name__, len(self), self._batch_size)

    def __getitem__(self, key):
        """Get a sample in the dataset using its indice.

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

        jobs = (delayed(self.__getitem__)(self._current_batch * self._batch_size + i) for i in range(batch_size))
        results = compute(jobs, get=threaded.get)[0]

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

        This methods returns the iterator to the first sample, and shuffle the
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
        random.shuffle(self.indices)

    def split(self, ratio):
        """Split the dataholder into two dataholders.

        The first one, will receive *total_samples * ratio* samples, and the second
        dataholder will receive the remaining samples.

        Args:
            ratio (float): ratio of split. Must be between 0 and 1.

        Returns:
            (data1, data2): two dataholder, in the same type that the original.

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
        """Split the dataholder into N dataholders with the sample number of samples each.

        Args:
            n (int): number of new splits.

        Returns:
            output (list): list of N dataholders.

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

    @property
    def total_samples(self):
        return self._total_samples

    @total_samples.setter
    def total_samples(self, total_samples):
        utils.assert_raise(total_samples >= 1, ValueError,
                           'number of samples must be greater or equal to 1')
        self._total_samples = total_samples
        self._remaining_samples = total_samples

    def __init__(self, *args, **kwargs):
        total_samples = kwargs.pop('total_samples', None)
        super(CallableHolder, self).__init__(*args, **kwargs)
        self._total_samples = total_samples

    def get_sample(self, key):
        return self._data(key)


class TensorHolder(AbsDataHolder):
    """
    A dataholder to work with :class:`torch.Tensor` objects.

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
        self._total_samples = len(self._data)

    def get_sample(self, key):
        return self._data[key]


def NumpyHolder(data, *args, **kwargs):
    """
    Works exactly like :class:`~cogitare.data.TensorHolder`.

    When creating the object, it converts the numpy data to Tensor using
    :func:`torch.from_numpy` and then creates an :class:`~cogitare.data.TensorHolder`
    instance.
    """
    data = torch.from_numpy(data)

    return TensorHolder(data=data, *args, **kwargs)


def AutoHolder(data, *args, **kwargs):
    """Check the data type to infer which dataholder to use.
    """
    if torch.is_tensor(data):
        return TensorHolder(data, *args, **kwargs)
    elif isinstance(data, numpy.ndarray):
        return NumpyHolder(data, *args, **kwargs)
    else:
        raise ValueError('Unable to infer data type!')
