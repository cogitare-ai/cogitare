from torch import nn
from cogitare.utils import not_training, training, StopTraining
from abc import ABCMeta, abstractmethod
from six import add_metaclass
from cogitare.feedbacks import LoggerFeedback, ProgressBarFeedback, PlottingFeedback
from cogitare import utils
from cogitare.core.plugin_interface import PluginInterface
from collections import OrderedDict


@add_metaclass(ABCMeta)
class Model(nn.Module):

    valid_hooks = ('on_start', 'on_start_epoch', 'on_start_batch',
                   'on_end_batch', 'on_end_epoch', 'on_end',
                   'before_backward', 'before_step')

    def __init__(self):
        super(Model, self).__init__()
        self.state = {}
        self.status = {}
        self.plugins = dict((name, OrderedDict()) for name in self.valid_hooks)
        self._requires_register_default = False

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def loss(self):
        pass

    def _register_default_plugins(self):
        self._requires_register_default = False
        self.register_plugin([
            LoggerFeedback(title='[%s]' % self.__class__.__name__),
            ProgressBarFeedback(total=self.state['max_epochs'], indice_name='current_epoch',
                                desc='epoch', leave=True),
            PlottingFeedback(data_source=['loss', 'validation_loss'],
                             max_epochs=self.state['max_epochs'])
        ], 'on_end_epoch')

        self.register_plugin([
            ProgressBarFeedback(total=self.state['num_batches'], indice_name='current_batch',
                                desc='batch', leave=True)
        ], 'on_end_batch')

    def register_default_plugins(self):
        self._requires_register_default = True

    def register_plugin(self, plugin, hook):
        utils.assert_raise(hook in self.valid_hooks, ValueError,
                           'Expected on of the following hooks: ' + ', '.join(self.valid_hooks))

        if not isinstance(plugin, list):
            plugin = [plugin]

        container = self.plugins[hook]

        for p in plugin:
            if not isinstance(p, PluginInterface):
                p = PluginInterface.from_function(p)

            if p.name in container:
                raise ValueError('A plugin with name "{}" already exists'.format(p.name))

            container[p.name] = p

    def hook(self, name):
        for key, plugin in self.plugins[name].items():
            status = plugin(**self.state)

            self.status[name + '_' + key] = status

    @training
    def learn(self, dataset, optimizer, validation_dataset=None, max_epochs=50):
        try:
            self.state = {
                'max_epochs': max_epochs,
                'num_batches': len(dataset),
                'model': self,
                'current_batch': 0,
                'current_epoch': 0,
                'sample': None,
                'output': None,
                'loss': None,
                'validation_loss': None
            }

            if self._requires_register_default:
                self._register_default_plugins()

            self.hook('on_start')

            for epoch in range(1, max_epochs + 1):
                self.state['current_epoch'] = epoch
                self.state['current_batch'] = 0
                self.state['sample'] = None
                self.state['output'] = None

                self.hook('on_start_epoch')
                total_loss = 0
                total_samples = 0

                for idx, sample in enumerate(dataset):
                    idx += 1

                    self.state['current_batch'] = idx
                    self.state['sample'] = sample
                    self.hook('on_start_batch')

                    optimizer.zero_grad()

                    output = self.forward(sample)
                    loss = self.loss(output, sample)

                    self.hook('before_backward')
                    loss.backward()
                    self.hook('before_step')
                    optimizer.step()

                    loss = loss.data[0]
                    total_loss += loss
                    total_samples += 1

                    self.state['loss'] = loss
                    self.state['output'] = output
                    self.hook('on_end_batch')

                total_loss /= total_samples

                self.state['current_batch'] = 0
                self.state['sample'] = None
                self.state['output'] = None
                self.state['loss'] = total_loss

                if validation_dataset:
                    self.state['validation_loss'] = self.evaluate(validation_dataset)

                self.hook('on_end_epoch')

            self.hook('on_end')
            return True
        except StopTraining:
            return False

    @not_training
    def predict(self, x):
        return self.forward(x)

    @not_training
    def evaluate(self, dataset):
        total_loss = 0

        for sample in dataset:
            output = self.forward(sample)
            loss = self.loss(output, sample)

            total_loss += loss.data[0]

        return total_loss / len(dataset)
