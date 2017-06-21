import math
import random
import numpy as np
import torch
from torch.utils import data
from cogitare import utils


class Dataset(object):

    @property
    def current_batch(self):
        return self._current_batch

    @property
    def total_samples(self):
        return self._total_samples

    @total_samples.setter
    def total_samples(self, value):
        if not callable(self.inputs) or not callable(self.targets):
            raise ValueError('You can set "total_samples" only when both inputs and'
                             ' outputs are callable')
        self._total_samples = value

    @property
    def remaining_samples(self):
        return self._remaining_samples

    def __init__(self, inputs, targets, batch_size=1, shuffle=True, drop_last=False):
        utils.assert_raise(batch_size >= 1, ValueError, '"batch_size" must be greater or equal to 1')
        self.inputs = self._to_dataset(inputs)
        self.targets = self._to_dataset(targets)
        self.batch_size = batch_size
        self._shuffle = shuffle
        self.drop_last = drop_last
        self._current_batch = 0
        self._total_samples = None
        self._remaining_samples = None
        self._indices = None
        self._requires_reset = False

        if isinstance(self.inputs, (list, torch.Tensor)) and \
           isinstance(self.targets, (list, torch.Tensor)):
            utils.assert_raise(len(self.inputs) == len(self.targets), ValueError,
                               '"inputs" and "outputs" must have the same length')

        tensor = None
        if not callable(inputs):
            tensor = inputs
        elif not callable(targets):
            tensor = targets

        if tensor is not None:
            self._total_samples = len(tensor)
            self._remaining_samples = len(tensor)
            indices = list(range(len(tensor)))
            self._indices = indices
            if shuffle:
                self.shuffle()

    @classmethod
    def from_dataloader(cls, dataloader, *args, **kwargs):
        return cls.from_dataset(dataloader.dataset, *args, **kwargs)

    @classmethod
    def from_dataset(cls, dataset, *args, **kwargs):
        if isinstance(dataset, data.TensorDataset):
            return cls(dataset.data_tensor, dataset.target_tensor, *args, **kwargs)
        else:
            raise ValueError('Only torch.utils.data.TansorDataset is supported')

    def _to_dataset(self, data):
        if isinstance(data, np.ndarray):
            return torch.from_numpy(data)
        elif isinstance(data, (list, torch.Tensor)) or callable(data):
            return data
        else:
            ValueError('Invalid data type {}. Must be a list, numpy.ndarray, torch.Tensor '
                       'or a callable that returns torch.Tensor'.format(type(data)))

    def __raise_if_callable(self, msg, both=True):
        if both:
            if callable(self.inputs) and callable(self.targets):
                raise ValueError(msg)
        else:
            if callable(self.inputs) or callable(self.targets):
                raise ValueError(msg)

    def __len__(self):
        self.__raise_if_callable('Unable to get the dataset length using '
                                 'callable data providers')
        tensor = self.inputs if not callable(self.inputs) else self.targets

        if self.drop_last:
            return len(tensor) // self.batch_size
        else:
            return (len(tensor) + self.batch_size - 1) // self.batch_size

    def reset(self):
        self._current_batch = 0
        self._remaining_samples = self.total_samples
        if self._shuffle:
            self.shuffle()

    def __iter__(self):
        if self._requires_reset:
            self.reset()
        return self

    def _get_batch(self):
        inputs = []
        targets = []
        batch_size = min(self.batch_size, self.remaining_samples)

        def _pos(i):
            return self.current_batch * self.batch_size + i

        if callable(self.inputs):
            for i in range(batch_size):
                x = self.inputs(_pos(i))
                if x is None and self.drop_last:
                    self._requires_reset = True
                    raise StopIteration
                inputs.append(x)
        else:
            for i in range(batch_size):
                x = self.inputs[self._indices[_pos(i)]]
                inputs.append(x)
            self._remaining_samples -= batch_size

        if callable(self.targets):
            for i in range(batch_size):
                y = self.targets(_pos(i))
                targets.append(y)
        else:
            for i in range(batch_size):
                y = self.targets[self._indices[_pos(i)]]
                targets.append(y)

        self._current_batch += 1
        return inputs, targets

    def __next__(self):
        if self.drop_last and self.remaining_samples < self.batch_size:
            self._requires_reset = True
            raise StopIteration
        if self.remaining_samples == 0:
            self._requires_reset = True
            raise StopIteration
        return self._get_batch()

    next = __next__  # python2

    def shuffle(self):
        if self.total_samples is None:
            raise ValueError('You must set the total_samples before shuffling')
        random.shuffle(self._indices)

    def split(self, ratio):
        if self.total_samples is None:
            raise ValueError('You must set the total_samples before splitting')
        utils.assert_raise(0 < ratio < 1, ValueError, '"ratio" must be between 0 and 1')

        pos = math.floor(len(self) * ratio)
        if callable(self.inputs):
            x1 = self.inputs
            x2 = self.inputs
        else:
            x1 = self.inputs[:pos]
            x2 = self.inputs[pos:]

        if callable(self.targets):
            y1 = self.targets
            y2 = self.targets
        else:
            y1 = self.targets[:pos]
            y2 = self.targets[pos:]

        dataset1 = Dataset(x1, y1, batch_size=self.batch_size, shuffle=self.shuffle, drop_last=self.drop_last)
        dataset2 = Dataset(x2, y2, batch_size=self.batch_size, shuffle=self.shuffle, drop_last=self.drop_last)

        if callable(self.inputs) and callable(self.targets):
            dataset1.total_samples = pos
            dataset2.total_samples = len(self) - pos

        dataset1._indices = self._indices[:pos]
        dataset2._indices = self._indices[pos:]

        return dataset1, dataset2

    def split_chunks(self, n):
        if self.total_samples is None:
            raise ValueError('You must set the total_samples before splitting')
        size = len(self) // n

        datasets = []

        for i in range(n):
            begin, end = i * size, min((i + 1) * size, size - (i + 1) * size)

            if callable(self.inputs):
                self.inputs
            else:
                x1 = self.inputs[begin:end]

            if callable(self.targets):
                self.targets
            else:
                y1 = self.targets[begin:end]

            dataset = Dataset(x1, y1, batch_size=self.batch_size, shuffle=self.shuffle, drop_last=self.drop_last)
            dataset._indices = self._indices[begin:end]

            if callable(self.inputs) and callable(self.targets):
                dataset.total_samples = end - begin
            datasets.append(dataset)

        return datasets
