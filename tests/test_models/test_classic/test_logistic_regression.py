from tests.common import TestCase
from cogitare.data import DataSet
import torch.optim as optim
import torch
import mock
import pytest
from cogitare.models.classic import LogisticRegression


class TestLogistic(TestCase):

    def test_create(self):
        with pytest.raises(ValueError) as info:
            LogisticRegression(0)
        self.assertIn('"input_size" value must be', str(info.value))

        with pytest.raises(ValueError) as info:
            LogisticRegression(10, num_classes=1)
        self.assertIn('"num_classes" must be', str(info.value))

        LogisticRegression(10, dropout=0)
        LogisticRegression(10, dropout=0.5)

        with pytest.raises(ValueError) as info:
            LogisticRegression(10, dropout=1.5)
        self.assertIn('"dropout" value must be', str(info.value))

        with mock.patch('cogitare.models.classic.LogisticRegression.cuda', autospec=True) as meth:
            LogisticRegression(10, use_cuda=True)

        assert meth.called

    def test_model(self):
        data = torch.DoubleTensor([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]])
        out = torch.LongTensor([0, 0, 0, 0])
        ds = DataSet([data, out])

        l = LogisticRegression(3)
        sgd = optim.SGD(l.parameters(), lr=0.001)

        self.assertEqual(l.learn(ds, sgd, max_epochs=10), True)
