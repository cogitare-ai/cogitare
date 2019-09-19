from cogitare import SequentialModel
import torch.optim as optim
import torch
import torch.nn.functional as F
import torch.nn as nn
from tests.common import TestCase
import cogitare
cogitare.seed(123)


class Model1(SequentialModel):

    def __init__(self):
        super(Model1, self).__init__()
        self.rnn = nn.RNNCell(1, 10)
        self.out = nn.Linear(10, 1)

    def get_initial_state(self, batch):
        super(Model1, self).get_initial_state(batch)
        h0 = torch.zeros(1, 10)

        return h0

    def forward(self, seq, prev_hidden, timestep, seqlen):
        super(Model1, self).forward(seq, prev_hidden, timestep, seqlen)
        input_data = seq[0]
        hidden = self.rnn(input_data, prev_hidden)

        output = self.out(hidden)

        return output, hidden

    def loss(self, output, sample, hidden, timestep, seqlen):
        super(Model1, self).loss(output, sample, hidden, timestep, seqlen)
        return F.mse_loss(output, sample[0], reduction='sum')


class Model2(Model1):

    def loss(self, output, sample, hidden, timestep, seqlen):
        if timestep != seqlen:
            return None

        super(Model1, self).loss(output, sample, hidden, timestep, seqlen)
        return F.mse_loss(output, sample[0], reduction='sum')


class TestSequentialModel(TestCase):

    data = [
        [
            (torch.Tensor([[1]]), torch.Tensor([[2]])),
            (torch.Tensor([[2]]), torch.Tensor([[3]])),
            (torch.Tensor([[3]]), torch.Tensor([[4]]))
        ],

        [
            (torch.Tensor([[0.5]]), torch.Tensor([[1.5]])),
            (torch.Tensor([[1.5]]), torch.Tensor([[2.5]])),
            (torch.Tensor([[2.5]]), torch.Tensor([[3.5]]))
        ]
    ]

    def test_model(self):
        model = Model1()

        model.forward_seq(self.data[0])

    def test_learn(self):
        model1 = Model1()
        model2 = Model1()

        sgd1 = optim.Adam(model1.parameters(), lr=0.01)
        sgd2 = optim.Adam(model2.parameters(), lr=0.01)

        model1.learn(self.data, sgd1, max_epochs=100)
        model2.learn(self.data, sgd2, max_epochs=100)

        assert sum(model1.evaluate(self.data)) < 0.01
        assert sum(model2.evaluate(self.data)) < 0.01
