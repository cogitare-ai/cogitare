from torch import nn
import torch
from cogitare.utils import not_training, training, StopTraining
from abc import ABCMeta, abstractmethod
from six import add_metaclass
from cogitare.plugins import Logger, ProgressBar, PlottingMatplotlib
from cogitare import utils
from cogitare.core.plugin_interface import PluginInterface
from collections import OrderedDict


@add_metaclass(ABCMeta)
class Model(nn.Module):
    """
    Model is an extension of :class:`torch.nn.Module` that includes support for
    training the model using the method :meth:`~cogitare.Model.learn`.

    While training, you can use plugins to watch and interact with the model.
    The plugin works like an event mechanism, you register a callback function to
    a specific event, and then you gain aceess to some variables of the model at
    specific steps of the training process.
    Check the :meth:`~cogitare.Model.register_plugin` for more information.
    """

    valid_hooks = ('on_start', 'on_start_epoch', 'on_start_batch',
                   'on_end_batch', 'on_end_epoch', 'on_end',
                   'before_backward', 'before_step', 'on_stop_training')

    def __init__(self):
        super(Model, self).__init__()
        self.state = {}
        self.status = {}
        self.plugins = dict((name, OrderedDict()) for name in self.valid_hooks)
        self._requires_register_default = False

    @abstractmethod
    def forward(self, data):
        """
        .. note:: When developing a Model, the class must implement this method.

        In the parameters, it'll receive the data obtained by the dataset iterator,
        and must return the model output after forwarding the data.

        Args:
            data: this is the data got from iterating over the dataset provided in the
                :meth:`~cogitare.Model.learn` method. Its type and shape depends exclusively on
                the input dataset, no transformations or type checking are made during training.
                For most models, this will be a tuple containing ``(x_data, y_data)``, but can be
                anything.

        Returns:
            output: the data after processing the input data. Usually this is a :class:`torch.autograd.Variable`.
        """
        pass

    @abstractmethod
    def loss(self, output, data):
        """
        .. note:: When developing a Model, the class must implement this method.

        It will receive the output of the :meth:`~cogitare.Model.forward` function and
        the data obtained by the dataset iterator (the same used in forward),
        and must return the model loss considering the model output and expected output.

        Args:
            output: the :meth:`~cogitare.Model.forward` output
            data: (note, this is the same data received in :meth:`~cogitare.Model.forward`)
                this is the data got from iterating over the dataset provided in the
                :meth:`~cogitare.Model.learn` method. Its type and shape depends exclusively on
                the input dataset, no transformations or type checking are made during training.
                For most models, this will be a tuple containing ``(x_data, y_data)``, but can be
                anything.
        """
        pass

    def _register_default_plugins(self):
        self._requires_register_default = False

        if self.state['validation_dataset'] is None:
            plot_data = 'train'
        else:
            plot_data = 'both'

        self.register_plugin([
            Logger(title='[%s]' % self.__class__.__name__),
            ProgressBar(),
            PlottingMatplotlib(source=plot_data)
        ], 'on_end_epoch')

        self.register_plugin([
            ProgressBar(monitor='batch')
        ], 'on_end_batch')

    def register_default_plugins(self):
        """
        This method register a set o common plugins to let you debug the model training.

        Plugins included:

            - Progress bar per batch and epoch
            - Plot training and validation losses
            - Log training loss

        If you want to have these plugins on training, just use this method before
        :meth:`~cogitare.Model.learn`.
        """
        self._requires_register_default = True

    def register_plugin(self, plugin, hook):
        """You can use this to register a plugin to an specific event of the model.

        You can register (hook) a plugin to some specific events that may occour
        during training:

            - **on_start**: executed when the model starts the training
              process. At this time, you have access to the following variables:

              - max_epochs
              - num_batches
              - model
              - current_batch (always 0)
              - current_epoch (always 0)
              - validation_dataset (if provided in the :meth:`~cogitare.Model.learn`).

            - **on_start_epoch**: executed when the model start the execution
              of a new epoch. At this time, you have access to the following variables:

              - max_epochs
              - num_batches
              - model
              - current_batch (always 0)
              - current_epoch
              - validation_dataset (if provided in the :meth:`~cogitare.Model.learn`).

            - **on_start_batch**: executed when the model start the execution
              of a new batch. At this time, you have access to the following variables:

              - max_epochs
              - num_batches
              - model
              - current_batch
              - current_epoch
              - sample
              - validation_dataset (if provided in the :meth:`~cogitare.Model.learn`).

            - **before_backward**: executed before the model backward the loss
              function. At this time, you have access to the following variables:

              - max_epochs
              - num_batches
              - model
              - current_batch
              - current_epoch
              - sample
              - loss
              - output
              - validation_dataset (if provided in the :meth:`~cogitare.Model.learn`).

            - **before_step**: executed before the model optimize the model
              parameters. At this time, you have access to the following variables:

              - max_epochs
              - num_batches
              - model
              - current_batch
              - current_epoch
              - sample
              - loss
              - output
              - validation_dataset (if provided in the :meth:`~cogitare.Model.learn`).

            - **on_end_batch**: executed when the model finishes the execution
              of a batch of data. At this time, you have access to the following variables:

              - max_epochs
              - num_batches
              - model
              - current_batch
              - current_epoch
              - sample
              - loss
              - output
              - validation_dataset (if provided in the :meth:`~cogitare.Model.learn`).

            - **on_end_epoch**: executed when the model finishes the execution
              of epoch (usually multiple batches). At this time, you have access to the following variables:

              - max_epochs
              - num_batches
              - model
              - current_batch (always 0)
              - current_epoch
              - loss
              - validation_dataset (if provided in the :meth:`~cogitare.Model.learn`).
              - validation_loss (if validation data is present)

            - **on_end**: executed when the model finishes the execution
              of all epoches. At this time, you have access to the following variables:

              - max_epochs
              - num_batches
              - model
              - current_batch (always 0)
              - current_epoch
              - loss
              - validation_dataset (if provided in the :meth:`~cogitare.Model.learn`).
              - validation_loss (if validation data is present)

            - **on_stop_training**: executed when a plugin raises a :exc:`cogitare.utils.StopTraining`.
              At this time, the variables acessible will depends on the training step that the
              exception occored.
        """
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
        """
        Optimize the model parameters using the dataset. This function use the algorithm::

            for epoch in max_epochs:
                for sample in data:
                    # forward the data
                    output = forward(sample)
                    error = loss(output, sample)

                    # optimize the parameters
                    backward(error)
                    optimizer.step()

        If the validation_dataset is present, it can be used by plugins to evaluate the
        validation/test loss/error during training.

        .. todo:: cite DataSet class
        """
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
                'validation_loss': None,
                'validation_dataset': validation_dataset
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

                for idx, sample in enumerate(dataset):
                    idx += 1

                    self.state['current_batch'] = idx
                    self.state['sample'] = sample
                    self.hook('on_start_batch')

                    optimizer.zero_grad()

                    output = self.forward(sample)
                    loss = self.loss(output, sample)

                    loss_value = loss.data[0]
                    total_loss += loss_value

                    self.state['loss'] = loss_value
                    self.state['output'] = output

                    self.hook('before_backward')
                    loss.backward()
                    self.hook('before_step')
                    optimizer.step()

                    self.hook('on_end_batch')

                total_loss /= len(dataset)

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
            self.hook('on_stop_training')
            self.hook('on_end')
            return False

    @not_training
    def predict(self, data):
        """
        Calls the forward on the provided data, but without affecting/using training
        variables.

        Args:
            data: batch data that will be used as the argument to call
              meth:`~cogitare.Model.forward`.

        Returns:
            output: the meth:`~cogitare.Model.forward` output.
        """
        return self.forward(data)

    @not_training
    def evaluate(self, dataset, averaged=True):
        """
        Iterate over batches in dataset and returns the averaged loss in all batches.

        This method does not affect training variables, and can be used to evaluate the
        model performance in a different data (such as validation and test sets).

        Args:
            dataset: batch iterator
            averaged (bool): if True, returns the averaged loss over the batches.

        Returns:
            output (float): the loss in the provided dataset.
        """
        total_loss = 0

        for sample in dataset:
            output = self.forward(sample)
            loss = self.loss(output, sample)

            total_loss += loss.data[0]

        if averaged:
            return total_loss / len(dataset)
        else:
            return total_loss

    def load(self, path):
        """Load the model parameters using :func:`torch.load` from a given path.

        Args:
            path: path of the serialized state_dict.
        """
        self.load_state_dict(torch.load(path))

    def save(self, path):
        """Save the model parameters using :func:`torch.save` to a given path.

        Args:
            path: path to save the model.
        """
        torch.save(self.state_dict(), path)
