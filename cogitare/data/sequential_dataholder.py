from cogitare.data.dataholder import AbsDataHolder
from six import add_metaclass
from six.moves import zip_longest
from abc import ABCMeta
from cogitare import utils
import torch
import numpy


@add_metaclass(ABCMeta)
class SequentialAbsDataHolder(AbsDataHolder):

    @property
    def padding_value(self):
        return self._padding_value

    @padding_value.setter
    def padding_value(self, value):
        self._padding_value = value

    def __init__(self, *args, padding_value=None, **kwargs):
        self._padding_value = padding_value

        super(SequentialAbsDataHolder, self).__init__(*args, **kwargs)

    def __iter__(self):
        return self

    def __next__(self):
        batch = super(SequentialAbsDataHolder, self).__next__()
        return list(zip_longest(*batch, fillvalue=self.padding_value))

    next = __next__


class SequentialCallableHolder(SequentialAbsDataHolder):

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
        super(SequentialCallableHolder, self).__init__(*args, **kwargs)
        self._total_samples = total_samples

    def get_sample(self, key):
        return self._data(key)


class SequentialTensorHolder(SequentialAbsDataHolder):

    def __init__(self, *args, **kwargs):
        super(SequentialTensorHolder, self).__init__(*args, **kwargs)
        self._total_samples = len(self._data)

    def get_sample(self, key):
        return self._data[key]


def SequentialNumpyHolder(data, *args, **kwargs):
    data = torch.from_numpy(data)

    return SequentialTensorHolder(data=data, *args, **kwargs)


def SequentialAutoHolder(data, *args, **kwargs):
    """Check the data type to infer which data holder to use.
    """
    if torch.is_tensor(data):
        return SequentialTensorHolder(data, *args, **kwargs)
    elif isinstance(data, numpy.ndarray):
        return SequentialNumpyHolder(data, *args, **kwargs)
    elif callable(data):
        return SequentialCallableHolder(data, *args, **kwargs)
    else:
        raise ValueError('Unable to infer data type!')
