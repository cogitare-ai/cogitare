import torch
import random
import math
from abc import ABCMeta, abstractmethod
from cogitare import utils
from six import add_metaclass
import numpy


@add_metaclass(ABCMeta)
class _DataHolder(object):

    @property
    def total_samples(self):
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
        return '{} with {}x{} samples'.format(type(self).__name__, len(self), self._batch_size)

    def __getitem__(self, key):
        utils.assert_raise(0 <= key < self.total_samples, IndexError, 'Invalid index')

        return self.get_sample(self.indices[key])

    def _get_batch(self):
        if self._requires_reset:
            self.reset()

        data = []

        batch_size = min(self._batch_size, self._remaining_samples)

        if batch_size < self._batch_size and self._drop_last:
            self._requires_reset = True
            raise StopIteration

        if batch_size == 0:
            self._requires_reset = True
            raise StopIteration

        for i in range(batch_size):
            data.append(self[self._current_batch * self._batch_size + i])

        self._current_batch += 1
        self._remaining_samples -= batch_size

        return data

    @abstractmethod
    def get_sample(self, key):
        pass

    def __len__(self):
        if self._drop_last:
            return self.total_samples // self._batch_size
        else:
            return (self.total_samples + self._batch_size - 1) // self._batch_size

    def __iter__(self):
        return self

    def __next__(self):
        return self._get_batch()

    next = __next__

    def reset(self):
        self._requires_reset = False
        self._current_batch = 0
        self._remaining_samples = self.total_samples
        if self._shuffle:
            self.shuffle()

    def shuffle(self):
        random.shuffle(self.indices)

    def split(self, ratio):
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


class CallableHolder(_DataHolder):

    @property
    def total_samples(self):
        utils.assert_raise(self._total_samples is None, ValueError,
                           'You must define the "total_samples" value')
        return self._total_samples

    @total_samples.setter
    def total_samples(self, total_samples):
        utils.assert_raise(total_samples >= 1, ValueError,
                           'number of samples must be greater or equal to 1')
        self._total_samples = total_samples
        self._remaining_samples = total_samples

    def __init__(self, *args, **kwargs):
        super(CallableHolder, self).__init__(*args, **kwargs)
        self._total_samples = None

    def get_sample(self, key):
        return self._data(key)


class TensorHolder(_DataHolder):

    def __init__(self, *args, **kwargs):
        super(TensorHolder, self).__init__(*args, **kwargs)
        self._total_samples = len(self._data)

    def get_sample(self, key):
        return self._data[key]


def NumpyHolder(data, *args, **kwargs):
    data = torch.from_numpy(data)

    return TensorHolder(data=data, *args, **kwargs)


def AutoHolder(data, *args, **kwargs):
    if torch.is_tensor(data):
        return TensorHolder(data, *args, **kwargs)
    elif isinstance(data, numpy.ndarray):
        return NumpyHolder(data, *args, **kwargs)
    else:
        raise ValueError('Unable to infer data type!')
