from cogitare.core.model import Model
import torch.nn.functional as F
import torch.nn as nn
from cogitare import utils


class LogisticRegression(Model):

    def __init__(self, input_size, num_classes=2, dropout=0, bias=True, *args, **kwargs):
        super(LogisticRegression, self).__init__(*args, **kwargs)
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

    def forward(self, x):
        x = x.view(x.size(0), -1)
        data = F.dropout(x, self.arguments['dropout'])
        out = self.linear(data)
        return F.log_softmax(out)

    def loss(self, output, expected):
        return F.nll_loss(output, expected)
