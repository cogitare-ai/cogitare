import torch
from cogitare.core.model import Model
import torch.nn.functional as F
import torch.nn as nn
from cogitare import utils
from torch.autograd import Variable


class LogisticRegression(Model):

    def __init__(self, input_size, num_classes=2, dropout=0, bias=True, use_cuda=False, *args, **kwargs):
        super(LogisticRegression, self).__init__(*args, **kwargs)
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
        x = Variable(utils.to_tensor(sample[0], torch.FloatTensor, self.use_cuda))
        x = x.view(x.size(0), -1)
        data = F.dropout(x, self.arguments['dropout'])
        out = self.linear(data)
        return F.log_softmax(out)

    def loss(self, output, sample):
        expected = Variable(utils.to_tensor(sample[1], torch.LongTensor, self.use_cuda))

        return F.nll_loss(output, expected)
