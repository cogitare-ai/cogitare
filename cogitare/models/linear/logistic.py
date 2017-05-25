from cogitare.core.model import Model
import torch.nn.functional as F
import torch.nn as nn


class LogisticRegression(Model):

    def __init__(self, input_size, num_classes=2, dropout=0, bias=True, *args, **kwargs):
        super(LogisticRegression, self).__init__(*args, **kwargs)
        self.arguments = {
            'input_size': input_size,
            'num_classes': num_classes,
            'dropout': dropout,
            'bias': bias,
        }

        self.linear = nn.Linear(input_size, num_classes, bias)

    def forward(self, x):
        data = F.dropout(x, self.arguments['dropout'])
        out = self.linear(data)
        return F.log_softmax(out)

    def loss(self, output, expected):
        return F.nll_loss(output, expected)
