from tests.common import TestCase
import mock
import functools
import numpy as np
import pytest
from cogitare.data import TensorHolder, DataSet
import torch


data1 = torch.rand((100, 32))
data2 = torch.rand((100, 32))


class TestDataSet(TestCase):

    def setUp(self):
        self.dh1 = TensorHolder(data1)
        self.dh2 = TensorHolder(data2)
        self.ds = DataSet([self.dh1, self.dh2], batch_size=5, shuffle=False)

    def test_create(self):
        ds1 = DataSet([self.dh1, self.dh2])
        ds2 = DataSet([data1, data2])

        for d1, d2 in zip(ds1.container, ds2.container):
            self.assertEqual(d1._data, d2._data)

    def test_share_indices(self):
        c = self.ds.container

        self.assertIs(c[0].indices, c[1].indices)

    def test_diff_size(self):
        dh1 = TensorHolder(data1)
        dh2 = TensorHolder(data1[0])

        with pytest.raises(ValueError) as info:
            DataSet([dh1, dh2])
        self.assertIn('All data must have the same length', str(info.value))

    def test_datatype(self):
        DataSet([self.dh1, self.dh2])
        DataSet((self.dh1, self.dh2))
        with pytest.raises(ValueError) as info:
            DataSet(self.dh1)
        self.assertIn('"data" must be a list or a tuple', str(info.value))

    def test_repr(self):
        r = repr(self.ds)
        self.assertIn('DataSet with:', r)
        self.assertIn('batch size: 5', r)
        self.assertIn('TensorHolder with 20x5 samples', r)

    def test_split(self):
        ds1, ds2 = self.ds.split(0.6)

        self.assertEqual(len(ds1), 12)
        self.assertEqual(len(ds2), 8)

        # check if sharing indices
        c1, c2 = ds1.container, ds2.container
        self.assertIs(c1[0].indices, c1[1].indices)
        self.assertIs(c2[0].indices, c2[1].indices)

        # check splited
        self.assertEqual(len(np.intersect1d(c1[0].indices, c2[0].indices)), 0)

    def test_split_chunks(self):
        dss = self.ds.split_chunks(10)

        for ds in dss:
            self.assertEqual(len(ds), 2)

        indices = [ds.container[0].indices for ds in dss]
        self.assertEqual(len(functools.reduce(np.intersect1d, indices)), 0)

    def test_shuffle(self):
        meth = self.ds.container[0].shuffle = mock.MagicMock()

        self.ds.shuffle()

        assert meth.called

    def test_reset(self):
        meths = []
        for dh in self.ds.container:
            m = dh.reset = mock.MagicMock(return_value=True)
            meths.append(m)

        self.ds.reset()
        assert all(m.called for m in meths)

    def test_len(self):
        c = self.ds.container
        for dh in c:
            self.assertEqual(len(dh), len(self.ds))

    def test_iter(self):
        for i in range(10):
            counter = 0
            for idx, batch in enumerate(self.ds):
                self.assertIsInstance(batch, list)
                self.assertEqual(len(batch), 2)
                counter += 1
            self.assertEqual(counter, len(self.ds))

    def test_getitem(self):
        for i in range(100):
            b1, b2 = self.ds[i]

            self.assertEqual(b1, data1[i])
            self.assertEqual(b2, data2[i])

    def test_set_total_samples(self):
        ds = DataSet([self.dh1, self.dh2], total_samples=13, batch_size=3)
        self.assertEqual(len(ds), 5)

        ds = DataSet([self.dh1, self.dh2], total_samples=13, batch_size=3, drop_last=True)
        self.assertEqual(len(ds), 4)
