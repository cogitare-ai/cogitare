from torch import nn
import humanize
import torch
from cogitare.utils import not_training, training, StopTraining
from cogitare.data import AbsDataHolder, SequentialAbsDataHolder, DataSet, AsyncDataLoader
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
    training the model using the method :meth:`~cogitare.Model.learn`, and that provides integration
    with plugins, model evalution metrics, and more.

    For models with sequential data (LSTM, GRU, RNNs, etcs), check the
    :class:`~cogitare.SequentialModel` model, that iterate over the timesteps per batch.

    While training, you can use plugins to watch and interact with the model.
    The plugin works like an event mechanism, you register a callback function to
    a specific event, and then you gain access to some variables of the model at
    specific steps of the training process.
    Check the :meth:`~cogitare.Model.register_plugin` for more information about the
    available events and variables that the model can interact with.

    Methods that your model must implement:

        - **forward** (data): where data is got from iterating over the dataset.
        - **loss** (output, data): where output is value returned from forward, and
          data is got from iterating over the dataset.

    Expected input on :meth:`~cogitare.Model.learn`:

        - **dataset** : an iterator, that returns one batch of samples per
          iteration. The batch can be of any type (list, numpy array, tensor, string, etcs).
          It is recommended to wrap your dataset using the :class:`~cogitare.data.DataSet` object,
          that provides a high-performance data loading interface.
    """

    valid_hooks = ('on_start', 'on_start_epoch', 'on_start_batch',
                   'on_end_batch', 'on_end_epoch', 'on_end',
                   'before_backward', 'before_step', 'on_stop_training')

    def __init__(self):
        super(Model, self).__init__()
        self._state = {}
        self._plugins = dict((name, OrderedDict()) for name in self.valid_hooks)
        self._requires_register_default = False
        self._logger = utils.get_logger(__name__)

    @abstractmethod
    def forward(self, data):
        """
        .. note:: When developing a Model, the class must implement this method.

        The method receive one parameter, the data obtained by the dataset iterator,
        and it must return the model output after forwarding the data.

        Args:
            data: this is the data got from iterating over the dataset provided in the
                :meth:`~cogitare.Model.learn` method. Its type and shape depend exclusively on
                the input dataset, no transformations or type checking are made during training.
                For most models, this will be a tuple containing ``(x_data, y_data)``, but can be
                anything.

        Returns:
            output: the data after processing the input data. Usually, this is a :class:`torch.autograd.Variable`.
        """
        pass

    @abstractmethod
    def loss(self, output, data):
        """
        .. note:: When developing a Model, the class must implement this method.

        It will receive the output of the :meth:`~cogitare.Model.forward` method and
        the data obtained by the dataset iterator (the same used in forward),
        and must return the model loss considering the model output and expected output.

        Args:
            output: the :meth:`~cogitare.Model.forward` output
            data: (note, this is the same data received in :meth:`~cogitare.Model.forward`)
                this is the data got from iterating over the dataset provided in the
                :meth:`~cogitare.Model.learn` method. Its type and shape depend exclusively on
                the input dataset, no transformations or type checking are made during training.
                For most models, this will be a tuple containing ``(x_data, y_data)``, but can be
                anything.

        Returns:
            loss (torch.autograd.Variable): the model loss. The loss will be used to backpropagate the errors.
        """
        pass

    def _register_default_plugins(self):
        self._requires_register_default = False

        plot = PlottingMatplotlib()
        plot.add_variable('loss_mean', 'Loss', color='blue', std_data='losses')
        if self._state['validation_dataset']:
            plot.add_variable('loss_mean_validation', 'Loss', color='green',
                              std_data='losses_validation')

        self.register_plugin([
            Logger(title='[%s]' % self.__class__.__name__),
            ProgressBar(),
            plot,
        ], 'on_end_epoch', True)

        self.register_plugin([
            ProgressBar(monitor='batch')
        ], 'on_end_batch', True)

    def register_default_plugins(self):
        """
        This method registers a set o common plugins to let you debug the model training.

        Plugins included:

            - Progress bar per batch and epoch
            - Plot training and validation losses
            - Log training loss

        If you want to have these plugins on training, just use this method before
        :meth:`~cogitare.Model.learn`.
        """
        self._requires_register_default = True

    def register_plugin(self, plugin, hook, override=False):
        """You can use this to register a plugin to a specific event of the model.

        You can register (hook) a plugin to some specific events that may occur
        during training:

            - **on_start**: executed when the model starts the training
              process. At this time, you have access to the following variables:

              - max_epochs
              - num_batches
              - model
              - optimizer
              - current_batch (always 0)
              - current_epoch (always 0)
              - validation_dataset (if provided in the :meth:`~cogitare.Model.learn`).

            - **on_start_epoch**: executed when the model starts the execution
              of a new epoch. At this time, you have access to the following variables:

              - max_epochs
              - num_batches
              - model
              - optimizer
              - current_batch (always 0)
              - current_epoch
              - validation_dataset (if provided in the :meth:`~cogitare.Model.learn`).

            - **on_start_batch**: executed when the model starts the execution
              of a new batch. At this time, you have access to the following variables:

              - max_epochs
              - num_batches
              - model
              - optimizer
              - current_batch
              - current_epoch
              - sample
              - validation_dataset (if provided in the :meth:`~cogitare.Model.learn`).

            - **before_backward**: executed before the model backward the loss
              function. At this time, you have access to the following variables:

              - max_epochs
              - num_batches
              - model
              - optimizer
              - current_batch
              - current_epoch
              - sample
              - losses (a list of all losses in the current epoch until now)
              - output
              - validation_dataset (if provided in the :meth:`~cogitare.Model.learn`).

            - **before_step**: executed before the model optimize the model
              parameters. At this time, you have access to the following variables:

              - max_epochs
              - num_batches
              - model
              - optimizer
              - current_batch
              - current_epoch
              - sample
              - losses (a list of all losses in the current epoch until now)
              - output
              - validation_dataset (if provided in the :meth:`~cogitare.Model.learn`).

            - **on_end_batch**: executed when the model finishes the execution
              of a batch of data. At this time, you have access to the following variables:

              - max_epochs
              - num_batches
              - model
              - optimizer
              - current_batch
              - current_epoch
              - sample
              - losses (a list of all losses in the current epoch until now)
              - output
              - validation_dataset (if provided in the :meth:`~cogitare.Model.learn`).

            - **on_end_epoch**: executed when the model finishes the execution
              of epoch (usually multiple batches). At this time, you have access to the following variables:

              - max_epochs
              - num_batches
              - model
              - optimizer
              - current_batch (always 0)
              - current_epoch
              - losses (a list of all losses in the current epoch until now)
              - loss_mean
              - losses_validation (if validation data is present. List of
                losses in all validation batches)
              - loss_mean_validation
              - validation_dataset (if provided in the :meth:`~cogitare.Model.learn`).

            - **on_end**: executed when the model finishes the execution
              of all epochs. At this time, you have access to the following variables:

              - max_epochs
              - num_batches
              - model
              - optimizer
              - current_batch (always 0)
              - current_epoch
              - losses (a list of all losses in the current epoch until now)
              - loss_mean
              - losses_validation (if validation data is present. List of
                losses in all validation batches)
              - loss_mean_validation
              - validation_dataset (if provided in the :meth:`~cogitare.Model.learn`).

            - **on_stop_training**: executed when a plugin raises a :exc:`cogitare.utils.StopTraining`.
              At this time, the variables accessible will depends on the training step that the
              exception occurred.

        Args:
            plugin (callable): a function to be called. The parameters will be sent
                was described above
            hook (str): the event to watch, as described above.
            override (bool): if True, override a plugin at a specific hook if it has the
                same name. If False, raises an exception.
        """
        utils.assert_raise(hook in self.valid_hooks, ValueError,
                           'Expected on of the following hooks: ' + ', '.join(self.valid_hooks))
        plugin = utils._ntuple(plugin, 1)

        container = self._plugins[hook]

        for p in plugin:
            if not isinstance(p, PluginInterface):
                p = PluginInterface.from_function(p)

            if p.name in container and not override:
                raise ValueError('A plugin with name "{}" already exists'.format(p.name))

            container[p.name] = p

    def hook(self, name):
        for key, plugin in self._plugins[name].items():
            status = plugin(**self._state)

            self._state[name + '_' + key] = status

    def _forward_batch(self, batch_num, batch, optimizer):
        optimizer.zero_grad()

        output = self.forward(batch)
        loss = self.loss(output, batch)

        self._state['output'] = output

        self.hook('before_backward')
        loss.backward()
        self.hook('before_step')
        optimizer.step()

        return loss.data[0]

    def _start_learn_state(self, dataset, optimizer, validation_dataset, max_epochs):
        self._logger.info('Model: \n\n{}\n'.format(repr(self)))
        if isinstance(dataset, (AbsDataHolder, SequentialAbsDataHolder,
                                DataSet, AsyncDataLoader)):
            self._logger.info('Training data: \n\n{}\n'.format(repr(dataset)))
        self._logger.info('Number of parameters: ' + humanize.intcomma(
            utils.number_parameters(self)))
        self._logger.info('Starting the training ...')

        self._state = {
            'max_epochs': max_epochs,
            'num_batches': len(dataset),
            'model': self,
            'optimizer': optimizer,
            'current_batch': 0,
            'current_epoch': 0,
            'sample': None,
            'output': None,
            'losses': None,
            'loss_mean': None,
            'losses_validation': None,
            'loss_mean_validation': None,
            'validation_dataset': validation_dataset
        }

        if self._requires_register_default:
            self._register_default_plugins()

    @training
    def learn(self, dataset, optimizer, validation_dataset=None, max_epochs=50):
        """
        Optimize the model parameters using the dataset. This function use the algorithm::

            for epoch in max_epochs:
                try:
                    for sample in data:
                        # forward the data
                        output = forward(sample)
                        error = loss(output, sample)

                        # optimize the parameters
                        backward(error)
                        optimizer.step()

                    if validation_dataset:
                        evaluate_model(validation_dataset)
                except StopTraining:
                    # stop the training process if request by a plugin

        If the ``validation_dataset`` is present, it can be used by plugins to evaluate the
        validation/test loss/error during training.

        To achieve a better performance, and have access to everyday dataset manipulation
        features, it's recommended to use the :class:`~cogitare.data.DataSet` class. It
        provides a interface that loads batches using multiple threads/processes
        and provides useful tasks such as data splitting, async data loading, shuffling, and more.

        Args:
            dataset (iterator): an iterator that returns one batch per iteration. To have a better
                performance and a easy to use interface, it is recommended to
                use the :class:`~cogitare.data.DataSet`.
            optimizer (torch.optim): the instance of a :class:`torch.optim.Optimizer` object.
            validation_dataset (iterator, optional): if provided, must have the same
                caracteristics that the ``dataset``. This may be used by the model and
                by plugins to evaluate the model performance during training.
            max_epochs (int): the number of epochs before ending the training procedure.

        Returns:
            status (bool): False if stopped by :class:`~cogitare.utils.StopTraining`. True otherwise.
        """
        original_register_default = self._requires_register_default
        self._start_learn_state(dataset, optimizer, validation_dataset, max_epochs)
        try:
            self.hook('on_start')

            for epoch in range(1, max_epochs + 1):
                self._state['current_epoch'] = epoch
                self._state['current_batch'] = 0
                self._state['sample'] = None
                self._state['output'] = None

                self.hook('on_start_epoch')
                losses = []
                self._state['losses'] = losses

                for idx, sample in enumerate(dataset, 1):
                    self._state['current_batch'] = idx
                    self._state['sample'] = sample
                    self.hook('on_start_batch')

                    loss = self._forward_batch(idx, sample, optimizer)
                    losses.append(loss)

                    self.hook('on_end_batch')

                self._state['current_batch'] = 0
                self._state['sample'] = None
                self._state['output'] = None
                self._state['loss_mean'] = sum(losses) / len(dataset)

                if validation_dataset:
                    val = self.evaluate(validation_dataset)
                    self._state['losses_validation'] = val
                    self._state['loss_mean_validation'] = sum(val) / len(validation_dataset)

                self.hook('on_end_epoch')

            status = True
        except StopTraining:
            self.hook('on_stop_training')
            status = False

        self._requires_register_default = original_register_default
        self._state.clear()
        self.hook('on_end')

        self._logger.info('Training finished')
        return status

    @not_training
    def predict(self, *args, **kwargs):
        """
        Calls the forward on the provided data, but without affecting/using training
        variables.

        Args:
            args/kwargs: :meth:`~cogitare.Model.forward` arguments. If provided, the
                forward will receive these parameters.

        Returns:
            output: the :meth:`~cogitare.Model.forward` output.
        """
        return self.forward(*args, **kwargs)

    @not_training
    def evaluate(self, dataset, *args, **kwargs):
        """
        Iterate over batches in the dataset and returns a list of the of losses of each batch.

        This method does not affect training variables and can be used to evaluate the
        model performance in a different data (such as validation and test sets).

        Args:
            dataset: batch iterator
            args/kwargs: :meth:`~cogitare.Model.forward` arguments. If provided, the
                forward will receive these parameters.

        Returns:
            output (list): the losses in the provided batches.
        """

        losses = []
        for sample in dataset:
            output = self.forward(sample)
            loss = self.loss(output, sample)

            losses.append(loss.data[0])

        return losses

    def load(self, path):
        """Load the model parameters using :func:`torch.load` from a given path.

        Args:
            path: path of the serialized state_dict.
        """
        state = torch.load(path)
        self.load_state_dict(state)

    def save(self, path):
        """Save the model parameters using :func:`torch.save` to a given path.

        Args:
            path: path to save the model.
        """
        state = self.state_dict()
        torch.save(state, path)
