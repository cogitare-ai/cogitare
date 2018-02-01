from cogitare import Model
import torch.optim as optim
from cogitare.core import PluginInterface
from cogitare.utils import StopTraining
import mock
import pytest
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from tests.common import TestCase
import cogitare
import os
cogitare.seed(123)


class Model1(Model):

    def __init__(self):
        super(Model1, self).__init__()
        self.p = nn.Parameter(torch.Tensor([0]))

    def forward(self, data):
        super(Model1, self).forward(data)
        return torch.mul(self.p, Variable(data[0]))

    def loss(self, output, data):
        super(Model1, self).loss(output, data)
        return F.mse_loss(output, Variable(data[1]), size_average=False)


class TestModel(TestCase):

    data = [(torch.Tensor([1, 2, 3]), torch.Tensor([2, 4, 6])),
            (torch.Tensor([2, 3, 4]), torch.Tensor([4, 6, 8]))]

    def test_model(self):
        model = Model1()

        output = model(self.data[0])
        loss = model.loss(output, self.data[0])
        output2 = model.predict(self.data[0])

        assert abs(loss.data[0] - 56) < 0.001
        self.assertEqual(output, output2)

        assert sum(model.evaluate(self.data[:1])) == 56

    def test_register_plugin_invalid_hook(self):
        model = Model1()

        with pytest.raises(ValueError) as info:
            model.register_plugin(lambda x: x, 'any', postpone=False)
        self.assertIn('Expected on of the following', str(info.value))

    def test_register_plugin_adds_to_hooks(self):
        model = Model1()

        def test(x):
            return x

        container = model._plugins['on_start']
        self.assertDictEqual(container, {})

        model.register_plugin(test, 'on_start', postpone=False)
        self.assertIn('test', container)

        plugin = container['test']

        self.assertEqual(plugin(3), 3)
        self.assertIsInstance(plugin, PluginInterface)

        with pytest.raises(ValueError) as info:
            model.register_plugin(test, 'on_start', postpone=False)
        self.assertIn('A plugin with name "test" already exists', str(info.value))

        model.register_plugin(test, 'on_start', True, postpone=False)
        self.assertIn('test', container)

        model.register_plugin(test, 'on_end', postpone=False)
        self.assertIn('test', model._plugins['on_end'])

    def test_hook(self):
        model = Model1()

        def test(*args, **kwargs):
            pass

        meth = mock.create_autospec(test, return_value=3)
        model.register_plugin(meth, 'on_start', postpone=False)

        model.hook('on_start')

        meth.assert_any_call()
        self.assertEqual(model.state['on_start_test'], 3)

    def test_register_default(self):
        model = Model1()
        sgd = optim.SGD(model.parameters(), lr=0.1)

        model.register_default_plugins()
        if 'DISPLAY' in os.environ:
            model.learn(self.data, sgd, self.data, max_epochs=10)
        else:
            model.learn(self.data, sgd, self.data, max_epochs=0)

        self.assertIn('Logger', model._plugins['on_end_epoch'])
        self.assertIn('ProgressBar', model._plugins['on_end_epoch'])
        self.assertIn('PlottingMatplotlib', model._plugins['on_end_epoch'])
        self.assertIn('ProgressBar', model._plugins['on_end_batch'])

    def test_evaluate_metrics(self):
        model = Model1()
        sgd = optim.SGD(model.parameters(), lr=0.01)
        model.learn(self.data, sgd, max_epochs=100)

        metrics = model.evaluate_with_metrics(self.data, {'loss': model.metric_loss,
                                                          'nothing': lambda x, y: 1})

        self.assertDictEqual(metrics, {'loss': [0, 0], 'nothing': [1, 1]})

    def test_metric_loss(self):
        model = Model1()

        output = model(self.data[0])
        loss = model.loss(output, self.data[0])

        self.assertEqual(loss.data[0], model.metric_loss(output, self.data[0]))

    @mock.patch('torch.load', return_value=None)
    def test_load(self, tload):
        model = Model1()
        model.load_state_dict = mock.MagicMock(return_value=None)

        model.load(__file__)

        tload.assert_called_with(__file__)

    @mock.patch('torch.save', return_value=None)
    def test_save(self, tsave):
        model = Model1()
        model.save(__file__)

    def test_stop_straining(self):
        def test(*args, **kwargs):
            raise StopTraining

        model = Model1()
        model.register_plugin(test, 'on_start')

        status = model.learn([1], None)

        self.assertEqual(status, False)

    def test_learn(self):
        model = Model1()
        sgd = optim.SGD(model.parameters(), lr=0.01)

        model.learn(self.data, sgd, max_epochs=50)

        assert sum(model.evaluate(self.data)) < 0.001

    def test_state(self):
        model1 = Model1()

        epochs = list(range(1, 51))
        batches = [1, 2] * 50

        def on_start(max_epochs, num_batches, model, **kw):
            self.assertEqual(max_epochs, 50)
            self.assertEqual(num_batches, 2)
            self.assertEqual(model, model1)

        def on_start_epoch(current_epoch, current_batch, **kw):
            self.assertEqual(current_batch, 0)
            self.assertEqual(current_epoch, epochs.pop(0))

        def on_start_batch(current_batch, sample, **kw):
            self.assertEqual(current_batch, batches.pop(0))
            self.assertEqual(sample, self.data[current_batch - 1])

        sgd = optim.SGD(model1.parameters(), lr=0.01)
        model1.register_plugin(on_start, 'on_start')
        model1.register_plugin(on_start_epoch, 'on_start_epoch')
        model1.register_plugin(on_start_batch, 'on_start_batch')

        model1.learn(self.data, sgd, self.data, max_epochs=50)
