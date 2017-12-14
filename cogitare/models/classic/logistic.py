import torch
from cogitare.core.model import Model
import torch.nn.functional as F
import torch.nn as nn
from cogitare import utils


class LogisticRegression(Model):
    """Implementation of the classic LogisticRegression model. If you want to use a model
    with hidden layers, consider using the :class:`~cogitare.models.feedforward.FeedForward`.

    Args:
        input_size (int): size of the input data. The input data must have the shape
            :math:`NxD`, where N is the batch size, and D is the ``input_size``.
        num_classes (int): number of output classes. A logistic regression will be created
            for each class.
        dropout (:obj:`float`): if defined, apply a dropout in the incoming data.
        bias (bool): if True, adds the bias term the linear layers.
        use_cuda (bool): if True, move the model parameters and data to cuda.
    """
    def __init__(self, input_size, num_classes=2, dropout=0.0, bias=True, use_cuda=False):
        super(LogisticRegression, self).__init__()
        self.use_cuda = use_cuda
        utils.assert_raise(num_classes >= 2, ValueError,
                           '"num_classes" must be greater than or equal 2')
        utils.assert_raise(0 <= dropout < 1, ValueError,
                           '"dropout" value must be between 0 and 1')
        utils.assert_raise(input_size >= 1, ValueError,
                           '"input_size" value must be greater than or equal 1')
        self.arguments = {
            'input_size': input_size,
            'num_classes': num_classes,
            'dropout': dropout,
            'bias': bias,
        }

        self.linear = nn.Linear(input_size, num_classes, bias)

        if use_cuda:
            self.cuda()

    def forward(self, sample):
        x = utils.to_variable(sample[0], use_cuda=self.use_cuda)
        x = x.view(x.size(0), -1)
        data = F.dropout(x, self.arguments['dropout'])
        out = self.linear(data)
        return F.log_softmax(out, dim=1)

    def loss(self, output, sample):
        expected = utils.to_variable(sample[1], torch.LongTensor, self.use_cuda)

        return F.nll_loss(output, expected)
