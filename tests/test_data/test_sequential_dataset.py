from tests.common import TestCase
from cogitare.data import SequentialTensorHolder, SequentialDataSet
import torch


data1 = torch.rand(100, 50, 32)
data2 = torch.rand(100, 50, 32)


class TestSequentialDataSet(TestCase):

    def setUp(self):
        self.dh1 = SequentialTensorHolder(data1)
        self.dh2 = SequentialTensorHolder(data2)
        self.ds = SequentialDataSet([self.dh1, self.dh2], batch_size=5, shuffle=False)

    def test_create(self):
        ds1 = SequentialDataSet([self.dh1, self.dh2])
        ds2 = SequentialDataSet([data1, data2])

        for d1, d2 in zip(ds1.container, ds2.container):
            self.assertIs(d1._data, d2._data)

    def test_set_padding(self):
        ds = SequentialDataSet([self.dh1, self.dh2], padding_value=123)

        for dh in ds.container:
            self.assertEqual(dh.padding_value, 123)

    def test_iter(self):
        for i in range(10):
            for batch in self.ds:
                self.assertEqual(len(batch), 50)
                for timestep in batch:
                    self.assertEqual(len(timestep), 2)
