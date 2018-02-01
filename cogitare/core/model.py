from torch import nn
import numpy as np
import humanize
import torch
from cogitare.utils import not_training, training, StopTraining
from cogitare.data import AbsDataHolder, SequentialAbsDataHolder, DataSet, AsyncDataLoader
from abc import ABCMeta, abstractmethod
from six import add_metaclass
from cogitare.plugins import Logger, ProgressBar, PlottingMatplotlib, Evaluator
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
        self.state = {}

        self._plugins = dict((name, OrderedDict()) for name in self.valid_hooks)
        self._requires_register_default = False
        self._logger = utils.get_logger(__name__)
        self._to_register = []

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

        .. note:: If you don't want that Cogitare backward the loss automatically, you must return None in this
            method. This should be useful if you want to do backprogragation yourself.

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
            loss (torch.autograd.Variable, None): the model loss. The loss will be used to backpropagate the errors.
        """
        pass

    def _register_default_plugins(self):
        self._requires_register_default = False

        plot = PlottingMatplotlib()
        plot.add_variable('loss', 'Loss', color='blue', use_std=True)
        if self.state['validation_dataset']:
            evaluator = Evaluator(self.state['validation_dataset'], {'loss': self.metric_loss})
            plot.add_variable('on_end_epoch_Evaluator_loss', 'Loss', color='green', use_std=True)
            self.register_plugin(evaluator, 'on_end_epoch', True, False)

        self.register_plugin([
            Logger(title='[%s]' % self.__class__.__name__),
            ProgressBar(),
            plot,
        ], 'on_end_epoch', True, False)

        self.register_plugin([
            ProgressBar(monitor='batch')
        ], 'on_end_batch', True, False)

    def register_default_plugins(self):
        """
        This method registers a set o common plugins to let you debug the model training.

        Plugins included:

            - Progress bar per batch and epoch
            - Plot training and validation losses (if validation_dataset is present)
            - Log training loss

        If you want to have these plugins on training, just use this method before
        :meth:`~cogitare.Model.learn`.
        """
        self._requires_register_default = True

    def register_plugin(self, plugin, hook, override=False, postpone=True):
        """You can use this to register a plugin to a specific event of the model.

        For each hook, the plugins will be called in the same order that they
        were registered.

        You can register (hook) a plugin to some specific events that may occur
        during training:

        +-------------------------+-------------+--------------+-------+-----------+----------------+----------------+---------+---------+---------+------------+---------------------+
        | hook X parameter        | max_epochs  | num_batches  | model | optimizer | current_batch  | current_epoch  | sample  | output  | loss    | loss_mean  | validation_dataset  |
        +=========================+=============+==============+=======+===========+================+================+=========+=========+=========+============+=====================+
        | on_start                | OK          | OK           | OK    | OK        | 0              | 0              | None    | None    | None    | None       | OK                  |
        +-------------------------+-------------+--------------+-------+-----------+----------------+----------------+---------+---------+---------+------------+---------------------+
        | on_start_epoch          | OK          | OK           | OK    | OK        | 0              | OK             | None    | None    | None    | None       | OK                  |
        +-------------------------+-------------+--------------+-------+-----------+----------------+----------------+---------+---------+---------+------------+---------------------+
        | on_start_batch          | OK          | OK           | OK    | OK        | OK             | OK             | OK      | None    | OK      | OK         | OK                  |
        +-------------------------+-------------+--------------+-------+-----------+----------------+----------------+---------+---------+---------+------------+---------------------+
        | before_backward         | OK          | OK           | OK    | OK        | OK             | OK             | OK      | OK      | OK      | None       | OK                  |
        +-------------------------+-------------+--------------+-------+-----------+----------------+----------------+---------+---------+---------+------------+---------------------+
        | before_step             | OK          | OK           | OK    | OK        | OK             | OK             | OK      | OK      | OK      | None       | OK                  |
        +-------------------------+-------------+--------------+-------+-----------+----------------+----------------+---------+---------+---------+------------+---------------------+
        | on_end_batch            | OK          | OK           | OK    | OK        | OK             | OK             | OK      | OK      | OK      | None       | OK                  |
        +-------------------------+-------------+--------------+-------+-----------+----------------+----------------+---------+---------+---------+------------+---------------------+
        | on_end_epoch            | OK          | OK           | OK    | OK        | 0              | OK             | None    | None    | OK      | OK         | OK                  |
        +-------------------------+-------------+--------------+-------+-----------+----------------+----------------+---------+---------+---------+------------+---------------------+
        | on_stop_training        | OK          | OK           | OK    | OK        | depends        | depends        | depends | depends | depends | depends    | OK                  |
        +-------------------------+-------------+--------------+-------+-----------+----------------+----------------+---------+---------+---------+------------+---------------------+
        | on_end                  | OK          | OK           | OK    | OK        | depends        | depends        | depends | depends | depends | depends    | OK                  |
        +-------------------------+-------------+--------------+-------+-----------+----------------+----------------+---------+---------+---------+------------+---------------------+

        The value of some of the model states depends on the execution of the model. The
        **on_stop_training**, for example, can be execute at any position of the learnining
        algorithm, so the valiables in the model state will depend on its current position.

        The value returned by the plugin is stored in model state as: ``hook + '_' + plugin_name``.
        If the return value is a dict, the state will have: ``hook + '_' + plugin_name + '_' + key`` for each key of the
        return key, and a ``hook + '_' + plugin_name + '__dict'`` with the whole dict. Where the ``plugin_name``
        if the function name, or the ``name`` attribute if using the :class:`~cogitare.core.PluginInterface`.

        For example, if your plugin named Sample returns ``{a: 3, b: 5, c: "cogitare"}``, another plugin can make use
        of this variables as follows::

            def my_plugin1(on_end_bach_Sample_a, on_end_bach_Sample_b, on_end_bach_Sample__dict, **kw):
                print(on_end_bach_Sample_a + on_end_bach_Sample_b)
                print(on_end_bach_Sample__dict)

        or::

            def my_plugin2(model, **kw):
                print(model.state['on_end_bach_Sample_a'] + model.state['on_end_bach_Sample_b'])
                print(model.state['on_end_bach_Sample__dict'])

        Args:
            plugin (callable): a function to be called. The parameters will be sent
                was described above
            hook (str): the event to watch, as described above.
            override (bool): if True, override a plugin at a specific hook if it has the
                same name. If False, raises an exception.
            postpone (bool): if True, creates the instance of the module only when
                starting the learning step, or when manually calling :meth:`~cogitare.Model.apply_register_plugins`.
        """  # noqa: E501
        if postpone:
            self._to_register.append((plugin, hook, override))
        else:
            self._apply_plugin(plugin, hook, override)

    def apply_register_plugins(self):
        """Creates and definetily register all plugins added by :meth:`~cogitare.Model.register_plugin` with
        `postpone=True`.

        This method is automatically executed when starting the :meth:`~cogitare.Model.learn`.
        """
        for params in self._to_register:
            self._apply_plugin(*params)
        self._to_register = []

    def _apply_plugin(self, plugin, hook, override):
        utils.assert_raise(hook in self.valid_hooks, ValueError,
                           'Invalid hook {}. Expected on of the following: {}'.format(
                               hook, ', '.join(self.valid_hooks)))
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
            status = plugin(**self.state)

            state_name = name + '_' + key
            if isinstance(status, dict):
                self.state[state_name + '__dict'] = status
                for attr, value in status.items():
                    self.state[state_name + '_' + attr] = value
            else:
                self.state[state_name] = status

    def _forward_batch(self, batch_num, batch, optimizer):
        optimizer.zero_grad()

        output = self(batch)
        loss = self.loss(output, batch)

        self.state['output'] = output

        self.hook('before_backward')
        if loss is not None and self.training:
            loss.backward()
            self.hook('before_step')
            optimizer.step()

            return loss.data[0]

    def _start_learn_state(self, dataset, optimizer, validation_dataset, max_epochs):
        self._logger.info('Model: \n\n{}\n'.format(repr(self)))
        if isinstance(dataset, (AbsDataHolder, SequentialAbsDataHolder,
                                DataSet, AsyncDataLoader)):
            self._logger.info('Training data: \n\n{}\n'.format(repr(dataset)))

        num_params = utils.number_parameters(self)
        self._logger.info('Number of trainable parameters: ' + humanize.intcomma(num_params[0]))
        self._logger.info('Number of non-trainable parameters: ' + humanize.intcomma(num_params[1]))
        self._logger.info('Total number of parameters: ' + humanize.intcomma(num_params[0] + num_params[1]))
        self._logger.info('Starting the training ...')

        self.state = dict(**{
            'max_epochs': max_epochs,
            'num_batches': len(dataset),
            'model': self,
            'optimizer': optimizer,
            'current_batch': 0,
            'current_epoch': 0,
            'sample': None,
            'output': None,
            'loss': None,
            'loss_mean': None,
            'validation_dataset': validation_dataset
        })

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
        self.apply_register_plugins()
        try:
            self.hook('on_start')

            for epoch in range(1, max_epochs + 1):
                self.state['current_epoch'] = epoch
                self.state['current_batch'] = 0
                self.state['loss'] = None
                self.state['loss_mean'] = None
                self.state['sample'] = None
                self.state['output'] = None

                self.hook('on_start_epoch')
                losses = []
                self.state['loss'] = losses

                for idx, sample in enumerate(dataset, 1):
                    self.state['current_batch'] = idx
                    self.state['sample'] = sample
                    self.state['output'] = None
                    self.hook('on_start_batch')

                    loss = self._forward_batch(idx, sample, optimizer)
                    losses.append(loss)

                    self.hook('on_end_batch')

                self.state['current_batch'] = 0
                self.state['sample'] = None
                self.state['output'] = None
                self.state['loss_mean'] = np.mean(self.state['loss'])

                self.hook('on_end_epoch')

            status = True
        except StopTraining:
            self.hook('on_stop_training')
            status = False
            self._logger.info('Training stopped')

        self._requires_register_default = original_register_default
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
        return self(*args, **kwargs)

    @not_training
    def metric_loss(self, output, sample, *args, **kwargs):
        """metric_loss is a shortcut to use the model loss as a tranining metric.

        Given the model output and the batch data, the metric_loss returns the
        loss for this specific batch.

        Args:
            output: model output
            sample: batch data

        Returns:
            output (float): the model loss.
        """
        loss = self.loss(output, sample, *args, **kwargs)
        return loss.data[0]

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
            output = self.predict(sample, *args, **kwargs)
            losses.append(self.metric_loss(output, sample, *args, **kwargs))

        return losses

    @not_training
    def evaluate_with_metrics(self, dataset, metrics, *args, **kwargs):
        """
        Iterate over batches in the dataset using metrics defined in the ``metrics``
        argument, and then return a dict mapping {matric_name -> list of results}.

        This method does not affect training variables and can be used to evaluate the
        model performance in a different data (such as validation and test sets).

        The ``metrics`` must be defined as:

            - key: a name for this metric. The metric name must follow variable naming convention.

            - value: a callable object, that accepts two parameters as input. The first parameter
              will be the model output, and the second parameter will be the batch data.

        Args:
            dataset: batch iterator
            metrics (dict): a dict mapping metric name to a callable.
            args/kwargs: :meth:`~cogitare.Model.forward` arguments. If provided, the
                forward will receive these parameters.

        Returns:

            output (dict): a dict mapping the metric name with a list containing the metric
            output for each batch in the dataset.


        Example::

            >>> metrics = {
            ...     'loss': model.metric_loss,
            ...     'precision': metrics.precision
            ... }

            >>> model.evaluate_with_metrics(validation_dataset, metrics)
            {'loss': [1.0, 0.8, 0.9], 'precision': [0.6, 0.55, 0.58]}
        """

        utils.assert_raise(isinstance(metrics, dict), ValueError,
                           '"metrics" must be a dict with metric_name -> metric_function')
        result = dict()

        for sample in dataset:
            output = self.predict(sample)

            for key, call in metrics.items():
                holder = result.get(key, list())
                holder.append(call(output, sample))

                result[key] = holder

        return result

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
