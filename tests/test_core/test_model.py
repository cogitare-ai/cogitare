import torch
import pytest
from cogitare import Model
from tests.common import TestCase


class Model1(Model):

    def forward(self, x):
        return x * 3

    def loss(self, x):
        pass


class TestModel(TestCase):

    def test_to_tensor(self):
        m = Model1(cuda=False)

        v = [1, 2, 3]
        self.assertEqual(m._to_tensor(v), torch.LongTensor(v))

        v = [1.0, 2.0, 3.0]
        self.assertEqual(m._to_tensor(v), torch.DoubleTensor(v))

        v = [torch.LongTensor([1, 2, 3]), torch.LongTensor([4, 5, 6])]
        vv = torch.LongTensor([[1, 2, 3], [4, 5, 6]])
        self.assertEqual(m._to_tensor(v), vv)

        v = ['as', 'cc']
        with pytest.raises(ValueError) as info:
            m._to_tensor(v)
        self.assertIn('Invalid data type:', str(info.value))

        v = [1, 2, 3]
        types = [torch.ByteTensor, torch.DoubleTensor, torch.FloatTensor, torch.LongTensor]

        for t in types:
            vv = t(v)
            self.assertEqual(type(m._to_tensor(v, t)), type(vv))
