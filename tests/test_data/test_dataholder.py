from tests.common import TestCase
import mock
from functools import reduce
import numpy as np
import pytest
import torch
from cogitare.data import AbsDataHolder, TensorHolder, NumpyHolder, CallableHolder, AutoHolder


_data = torch.rand((100, 32))


def get_data(idx):
    return _data[idx]


class _DataHolderAbs(object):

    def test_create(self):
        self.holder(self.data, **self.kwargs)
        with pytest.raises(ValueError) as info:
            self.holder(self.data, mode='asd')
        self.assertIn('"mode" must be one of:', str(info.value))

    def test_repr(self):
        dh = self.holder(self.data, **self.kwargs)
        out = repr(dh)
        self.assertIn('100x1 samples', out)

        dh = self.holder(self.data, batch_size=5, **self.kwargs)
        out = repr(dh)
        self.assertIn('20x5 samples', out)

    def test_getitem(self):
        dh = self.holder(self.data, batch_size=5, **self.kwargs)

        for i in range(10):
            self.assertEqual(dh[i], self._data[i])

    def test_on_sample_loaded(self):
        def f(x):
            return x * 2

        dh = self.holder(self.data, on_sample_loaded=f, **self.kwargs)

        for i in range(10):
            self.assertEqual(dh[i], self._data[i] * 2)

    def test_len(self):
        dh = self.holder(self.data, batch_size=9, **self.kwargs)
        self.assertEqual(12, len(dh))

        dh = self.holder(self.data, batch_size=9, drop_last=True, **self.kwargs)
        self.assertEqual(11, len(dh))

    def test_split(self):
        dh = self.holder(self.data, batch_size=9, **self.kwargs)
        self.assertEqual(12, len(dh))

        dh1, dh2 = dh.split(0.6)
        self.assertEqual(dh1.total_samples, 60)
        self.assertEqual(dh2.total_samples, 40)

        self.assertEqual(len(dh1), 7)
        self.assertEqual(len(dh2), 5)

        self.assertEqual(len(np.intersect1d(dh1.indices,  dh2.indices)), 0)

    def test_split_chunks(self):
        dh = self.holder(self.data, batch_size=10, **self.kwargs)
        self.assertEqual(10, len(dh))

        holders = dh.split_chunks(10)
        for holder in holders:
            self.assertEqual(holder.total_samples, 10)
            self.assertEqual(len(holder), 1)

        indices = [holders[i].indices for i in range(10)]
        self.assertEqual(len(reduce(np.intersect1d, indices)), 0)

    def test_shuffle(self):
        dh = self.holder(self.data, batch_size=10, **self.kwargs)
        dh.shuffle = mock.MagicMock(return_value=None)
        next(dh)
        assert dh.shuffle.called

        dh = self.holder(self.data, batch_size=10, shuffle=False, **self.kwargs)
        dh.shuffle = mock.MagicMock(return_value=None)
        next(dh)
        assert not dh.shuffle.called

    def test_batch(self):
        dh = self.holder(self.data, batch_size=10, **self.kwargs)
        self.assertEqual(dh._current_batch, 0)
        next(dh)
        self.assertEqual(dh._current_batch, 1)
        next(dh)
        self.assertEqual(dh._current_batch, 2)
        dh.reset()
        self.assertEqual(dh._current_batch, 0)
        next(dh)
        self.assertEqual(dh._current_batch, 1)
        dh.batch_size = 3
        next(dh)
        self.assertEqual(dh._current_batch, 1)
        self.assertEqual(dh.batch_size, dh._batch_size)

    def test_set_total_samples(self):
        dh = self.holder(self.data, batch_size=10, **self.kwargs)
        dh.total_samples = 90
        self.assertEqual(len(dh), 9)

        dh.total_samples = 100
        self.assertEqual(len(dh), 10)

        if self.holder != CallableHolder:
            with pytest.raises(ValueError) as info:
                dh.total_samples = 110
            self.assertIn('The value must be lesser or equal to the', str(info.value))

        with pytest.raises(ValueError) as info:
            dh.total_samples = 0
        self.assertIn('number of samples must be greater or equal to 1', str(info.value))

    def test_on_batch_loaded(self):
        def f(batch):
            return batch[2]

        dh = self.holder(self.data, shuffle=False, batch_size=10, on_batch_loaded=f, **self.kwargs)
        batch = next(dh)

        self.assertEqual(batch, self._data[2])

    def test_iter_batches(self):
        for mode in ['threaded', 'multiprocessing', 'sequential']:
            dh = self.holder(self.data, batch_size=10, mode=mode, **self.kwargs)
            for i in range(5):
                for batch in dh:
                    self.assertEqual(len(batch), 10)

        dh = self.holder(self.data, batch_size=8, drop_last=True, **self.kwargs)
        self.assertEqual(len(dh), 12)
        for batch in dh:
            self.assertEqual(len(batch), 8)

    def test_iter_single(self):
        dh = self.holder(self.data, batch_size=1, **self.kwargs)
        for batch in dh:
            self.assertIsInstance(batch, list)
            self.assertEqual(len(batch), 1)

        dh = self.holder(self.data, batch_size=1, single=True, **self.kwargs)
        for batch in dh:
            assert torch.is_tensor(batch)


class TestAbsInterface(TestCase):

    def test_get_sample(self):
        class A(AbsDataHolder):

            def get_sample(self, key):
                return super(A, self).get_sample(key)

        a = A(None)
        a.get_sample(0)


class TestTensorHolder(TestCase, _DataHolderAbs):

    data = torch.rand((100, 32))
    _data = data
    holder = TensorHolder
    kwargs = {}


class TestNumpyHolder(TestCase, _DataHolderAbs):

    @property
    def holder(self):
        return NumpyHolder

    data = np.random.rand(100, 32)
    _data = data
    kwargs = {}


class TestCallableHolder(TestCase, _DataHolderAbs):

    @property
    def holder(self):
        return CallableHolder

    kwargs = {'total_samples': 100}
    _data = _data

    @property
    def data(self):
        return get_data

    def test_not_total_samples(self):
        dh = CallableHolder(self.data)

        with pytest.raises(ValueError) as info:
            dh.total_samples
        self.assertIn('"total_samples" not defined', str(info.value))


class TestAutoHolder(TestCase):

    def test_tensor(self):
        dh = AutoHolder(_data)
        self.assertIsInstance(dh, TensorHolder)

    def test_numpy(self):
        dh = AutoHolder(np.random.rand(3, 3))
        self.assertIsInstance(dh, TensorHolder)

    def test_callable(self):
        dh = AutoHolder(get_data)
        self.assertIsInstance(dh, CallableHolder)

    def test_unknown(self):
        with pytest.raises(ValueError) as info:
            AutoHolder('asd')
        self.assertIn('Unable to infer data type', str(info.value))
