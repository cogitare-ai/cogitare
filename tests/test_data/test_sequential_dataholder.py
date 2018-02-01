from tests.common import TestCase
import pytest
import torch
import numpy as np
from cogitare.data import SequentialNumpyHolder,\
        SequentialTensorHolder, SequentialCallableHolder, SequentialAutoHolder


_data = torch.rand(100, 50, 32)


def get_data(idx):
    return _data[idx]


class _SequentialDataHolderAbs(object):

    def test_create(self):
        dh = self.holder(self.data, **self.kwargs)
        self.assertEqual(len(dh), 100)
        dh.padding_value = '_PAD_'

    def test_iter(self):
        dh = self.holder(self.data, batch_size=10, **self.kwargs)
        for batch in dh:
            self.assertEqual(len(batch), 50)
            self.assertEqual(len(batch[0]), 10)

            for timestep in batch:
                self.assertEqual(len(timestep), 10)


class TestTensorHolder(TestCase, _SequentialDataHolderAbs):

    data = _data
    _data = _data
    holder = SequentialTensorHolder
    kwargs = {}


class TestNumpyHolder(TestCase, _SequentialDataHolderAbs):

    data = np.random.rand(100, 50, 32)
    data = data
    kwargs = {}

    @property
    def holder(self):
        return SequentialNumpyHolder


class TestCallableHolder(TestCase, _SequentialDataHolderAbs):

    _data = _data
    kwargs = {'total_samples': 100}

    @property
    def holder(self):
        return SequentialCallableHolder

    @property
    def data(self):
        return get_data

    def test_sort(self):
        samples = [
            [['a'], ['b'], ['c'], ['d']],
            [['e'], ['f']],
            [['g'], ['h'], ['i']]
        ]

        def load(pos):
            return samples[pos]

        dh = SequentialCallableHolder(load, batch_size=3, sort_by_len=True, total_samples=3,
                                      padding_value=33, shuffle=False)
        data, indices, sizes = next(dh)

        self.assertListEqual(sizes, [4, 3, 2])
        self.assertListEqual(indices, [0, 2, 1])
        self.assertListEqual(data, [(['a'], ['g'], ['e']),
                                    (['b'], ['h'], ['f']),
                                    (['c'], ['i'], 33),
                                    (['d'], 33, 33)])


class TestSequentialAutorHolder(TestCase):

    def test_tensor(self):
        dh = SequentialAutoHolder(_data)
        self.assertIsInstance(dh, SequentialTensorHolder)

    def test_numpy(self):
        dh = SequentialAutoHolder(np.random.rand(100, 50, 32))
        self.assertIsInstance(dh, SequentialTensorHolder)

    def test_callable(self):
        dh = SequentialAutoHolder(get_data)
        self.assertIsInstance(dh, SequentialCallableHolder)

    def test_unknown(self):
        with pytest.raises(ValueError) as info:
            SequentialAutoHolder('aa')
        self.assertIn('Unable to infer data type', str(info.value))
