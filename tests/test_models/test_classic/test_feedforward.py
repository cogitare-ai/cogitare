from tests.common import TestCase
from cogitare.models.classic import FeedForward
import torch.nn as nn
from cogitare.data import DataSet
import torch.optim as optim
import torch
import mock
import pytest


class TestFeedforward(TestCase):

    def test_create(self):
        with pytest.raises(ValueError) as info:
            FeedForward(0, 0, 0)
        self.assertIn('"input_size" must be', str(info.value))

        with pytest.raises(ValueError) as info:
            FeedForward(10, 0, 0)
        self.assertIn('"num_layers" must be', str(info.value))

        FeedForward(10, 15, 20)
        FeedForward(10, 15, 20, activation=nn.Tanh(), in_dropout=0.3, hidden_dropout=0.3,
                    loss_function=nn.LogSigmoid(), bias=False)

        with pytest.raises(ValueError) as info:
            FeedForward(10, 10, [1, 2])
        self.assertIn('"hidden_size" must have', str(info.value))

        FeedForward(10, 15, 20, nn.Tanh())
        FeedForward(10, 15, 20, [nn.Tanh()] * 15)
        with pytest.raises(ValueError) as info:
            FeedForward(10, 15, 20, [nn.Tanh()] * 10)
        self.assertIn('"activation" must have ', str(info.value))

        FeedForward(10, 15, 20, hidden_dropout=0.5)
        FeedForward(10, 15, 20, hidden_dropout=[0.5] * 15)
        with pytest.raises(ValueError) as info:
            FeedForward(10, 15, 20, hidden_dropout=[0.5] * 11)
        self.assertIn('"hidden_dropout" must have ', str(info.value))

        FeedForward(10, 15, 20, bias=False)
        FeedForward(10, 15, 20, bias=[False] * 15)
        with pytest.raises(ValueError) as info:
            FeedForward(10, 15, 20, bias=[False] * 10)
        self.assertIn('"bias" must have ', str(info.value))

        with mock.patch('cogitare.models.classic.FeedForward.cuda', autospec=True) as meth:
            FeedForward(10, 15, 20, use_cuda=True)

        assert meth.called

    def test_model(self):
        data = torch.DoubleTensor([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]])
        out = torch.LongTensor([0, 0, 0, 0])
        ds = DataSet([data, out])

        l = FeedForward(3, 2, 5)
        sgd = optim.SGD(l.parameters(), lr=0.001)

        self.assertEqual(l.learn(ds, sgd, max_epochs=10), True)
