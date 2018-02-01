import torch.nn as nn
from cogitare import utils
from cogitare.core.model import Model


class FeedForward(Model):
    """Implementation of the classic FeedForward (a.k.a. MLP - Multi-Layer Perceptron).
    This is a sequence of Linear layers, with dropout and activation functions in between.

    Args:
        input_size (int): size of the input data. The input data must have the shape
            :math:`NxD`, where N is the batch size, and D is the ``input_size``.
        num_layers (int): number of hidden layers in the model. For each layer, you
            can define a dropout value to apply in its input, and its activation function.
        hidden_size (int, tuple, list): if int, the dimension of the hidden layer for all layers.
            If a list or a tuple, you can define de hidden layer size for each hidden layer.
        activation (torch.nn.Module): a PyTorch activation function. By default, it uses tanh in all layers but
            the last one, which uses a log-softmax.
        in_dropout (:obj:`float`): if defined, the dropout in the input data.
        hidden_dropout (:obj:`float`, tuple, list): dropout to the input of the hidden layer. If is a float,
            all hidden layers will have the same dropout applyed to the incoming data. If a list or a tuple,
            you can define custom dropout values per layer.
        loss_function (torch.nn.Module): feed-forward loss function. By default, it uses the negative log-likelihood.
        bias (bool, tuple, list): if True (the default), adds the bias term in the linear layer. If tuple or a list,
            the bias term can be specified per layer.
        use_cuda (bool): if True, move the model parameters and data to cuda.
    """

    def __init__(self, input_size, num_layers, hidden_size, activation=None, in_dropout=0.0,
                 hidden_dropout=0.0, loss_function=None, bias=True, use_cuda=None):
        super(FeedForward, self).__init__()
        if activation is None:
            activation = [nn.Tanh()] * (num_layers - 1)
            activation.append(nn.LogSoftmax(dim=1))
        if loss_function is None:
            loss_function = nn.NLLLoss()

        utils.assert_raise(input_size >= 1, ValueError, '"input_size" must be greater or equal to 1')
        utils.assert_raise(num_layers >= 1, ValueError, '"num_layers" must be greater or equal to 1')

        self.input_size = input_size
        self.num_layers = num_layers
        self.in_dropout = in_dropout
        self.loss_function = loss_function
        self.use_cuda = use_cuda

        self.hidden_size = utils._ntuple(hidden_size, num_layers)
        self.activation = utils._ntuple(activation, num_layers)
        self.hidden_dropout = utils._ntuple(hidden_dropout, num_layers)
        self.bias = utils._ntuple(bias, num_layers)

        # make some assertions before continuing
        utils.assert_raise(len(self.hidden_size) == num_layers, ValueError,
                           '"hidden_size" must have the same length that "num_layers"')
        utils.assert_raise(len(self.activation) == num_layers, ValueError,
                           '"activation" must have the same length that "num_layers"')
        utils.assert_raise(len(self.hidden_dropout) == num_layers, ValueError,
                           '"hidden_dropout" must have the same length that "num_layers"')
        utils.assert_raise(len(self.bias) == num_layers, ValueError,
                           '"bias" must have the same length that "num_layers"')

        self._mlp = self._make_model()

        if use_cuda:
            self.cuda()

    def _make_model(self):
        layers = []
        if self.in_dropout:
            layers.append(nn.Dropout(self.in_dropout))

        in_size = self.input_size
        for size, activation, dropout, bias in zip(self.hidden_size, self.activation, self.hidden_dropout, self.bias):
            linear = nn.Linear(in_size, size, bias)
            in_size = size

            if dropout:
                layers.append(nn.Dropout(dropout))

            layers.append(linear)
            layers.append(activation)

        return nn.Sequential(*layers)

    def forward(self, sample):
        x = utils.to_variable(sample[0], use_cuda=self.use_cuda)
        x = x.view(x.size(0), -1)

        return self._mlp(x)

    def loss(self, output, sample):
        expected = utils.to_variable(sample[1], use_cuda=self.use_cuda)
        return self.loss_function(output, expected)
