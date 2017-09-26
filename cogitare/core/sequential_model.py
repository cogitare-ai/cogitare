from abc import ABCMeta, abstractmethod
from six import add_metaclass
from cogitare.core.model import Model
from cogitare.utils import not_training, training


@add_metaclass(ABCMeta)
class SequentialModel(Model):
    """
    SequentialModel is an extension of :class:`~cogitare.Model` that includes support for sequential
    models. It's designed to work with RNNs, such as LSTM and GRUs, and can be easily used for any
    model that operates over timestep per timestep.

    If you are using a RNN, but passing the whole sequence as input, you should consider using
    the :class:`~cogitare.Model` interface. This interface is desined for timestep per timestep
    and can be used for Many-to-Many models and for Many-to-One.

    While training, you can use plugins to watch and interact with the model.
    The plugin works like an event mechanism, you register a callback function to
    a specific event, and then you gain access to some variables of the model at
    specific steps of the training process.
    Check the :meth:`~cogitare.Model.register_plugin` for more information about the
    available events and variables that the model can interact with.

    Methods that your model must implement:

        - **forward** (batch, hidden, timestep, seqlen): receives the data at
            the current timestep, the hidden state, current timestep, and the sequence size;
        - **loss** (output, batch, hidden, timestep, seqlen): returns the loss
            at the current timestep;
        - **get_initial_state** (self, batch): start the RNN hidden state.

    Expected input on :meth:`~cogitare.Model.learn`:

        - **dataset** : an iterator, that returns one batch of samples per
          iteration. The batch can be of any type (list, numpy array, tensor, string, etcs).
          It is recommended to wrap your dataset using the :class:`~cogitare.data.DataSet` object,
          that provides a high-performance data loading interface.
    """

    def __init__(self):
        self.valid_hooks = self.valid_hooks + ('on_start_timestep', 'on_end_timestep')
        super(SequentialModel, self).__init__()

    @abstractmethod
    def get_initial_state(self, batch):
        """Returns the initial state of the RNN.

        Args:
            batch: the current batch.

        Returns:
            state (torch.autograd.Variable): the initial state.
        """
        pass

    @abstractmethod
    def forward(self, batch, hidden, timestep, seqlen):
        """
        .. note:: When developing a Model, the class must implement this method.

        The method receive four parameters, the data obtained by the timestep iterator,
        the hidden state at the current timestep, the timestep, and the leghth of the sequence.

        It must return a tuple with the model output after forwarding the data and the new hidden state.

        Args:
            batch: this is the data got from iterating over the timesteps, got from
                iterating over the batches in the dataset provided in the
                :meth:`~cogitare.Model.learn` method. Its type and shape depend exclusively on
                the input dataset, no transformations or type checking are made during training.
                For most models, this will be a tuple containing ``(x_data_t, y_data_t)``, but can be
                anything.
            hidden (torch.autograd.Variable): the hidden state at the current timestep. If this is the first timestep,
                the hidden state is got from :meth:`~cogitare.SequentialModel.get_initial_state`. Otherwise, it is got
                from the :meth:`~cogitare.SequentialModel.forward` returned value.
            timestep (int): indicates the current timestem (from 1 to seqlen)
            seqlen (int): the number of timesteps in the sequence.

        Returns:
            (output, hidden): the data after processing the input data, and the new hidden state.

            Usually, these are :class:`torch.autograd.Variable`.
        """
        pass

    @abstractmethod
    def loss(self, output, batch, hidden, timestep, seqlen):
        """
        .. note:: When developing a Model, the class must implement this method.

        It will receive the output and the hidden state of the :meth:`~cogitare.Model.forward` method,
        with the the data obtained by the timestep iterator (the same used in forward),
        and must return the model loss considering the model output and expected output.

        If the model is Many-to-Many, it should return a valid loss for each timestep.

        If the model is Many-to-One, it should return a valid loss in the last timestep (
        when timestep == seqlen), and return None otherwise.

        Args:
            output: the :meth:`~cogitare.SequentialModel.forward` output
            batch: this is the data got from iterating over the timesteps, got from
                iterating over the batches in the dataset provided in the
                :meth:`~cogitare.Model.learn` method. Its type and shape depend exclusively on
                the input dataset, no transformations or type checking are made during training.
                For most models, this will be a tuple containing ``(x_data_t, y_data_t)``, but can be
                anything.
            hidden (torch.autograd.Variable): the hidden state at the current timestep. If this is the first timestep,
                the hidden state is got from :meth:`~cogitare.SequentialModel.get_initial_state`. Otherwise, it is got
                from the :meth:`~cogitare.SequentialModel.forward` returned value.
            timestep (int): indicates the current timestem (from 1 to seqlen)
            seqlen (int): the number of timesteps in the sequence.

        Returns:
            loss (torch.autograd.Variable, None): the model loss. The loss will be used to backpropagate the errors.
        """
        pass

    def _forward_batch(self, batch_num, batch, optimizer):
        seqlen = len(batch)
        losses = []
        total_loss = 0
        self._state['num_timesteps'] = seqlen
        self._state['losses_timestep'] = losses
        self._state['current_timestep'] = None

        optimizer.zero_grad()
        hidden = self.get_initial_state(batch)

        for timestep, sample in enumerate(batch, 1):
            self._state['current_timestep'] = timestep
            self._state['sample_at_timestep'] = sample
            self.hook('on_start_timestep')

            output, hidden = self.forward(sample, hidden, timestep, seqlen)

            loss = self.loss(output, sample, batch, hidden, timestep, seqlen)

            if loss is not None:
                total_loss += loss
                losses.append(loss.data[0])

            self._state['output_at_timestep'] = output
            self.hook('on_end_timestep')

        self._state['output'] = output

        self.hook('before_backward')
        total_loss.backward()
        self.hook('before_step')
        optimizer.step()

        return sum(losses) / len(losses)

    @not_training
    def evaluate(self, dataset, *args, **kwargs):
        """
        Iterate over batches in the dataset and returns a list of the of losses of each batch.

        This method does not affect training variables and can be used to evaluate the
        model performance in a different data (such as validation and test sets).

        Args:
            dataset: batch-timestep iterator
            args/kwargs: :meth:`~cogitare.SequentialModel.forward` arguments. If provided, the
                forward will receive these parameters.

        Returns:
            output (list): the losses in the provided batches, one loss per batch.
        """
        losses = []
        for batch in dataset:

            hidden = self.get_initial_state(batch)
            seqlen = len(batch)
            losses_batch = []

            for timestep, sample in enumerate(batch, 1):
                output, hidden = self.forward(sample, hidden, timestep, seqlen)
                loss = self.loss(output, sample, batch, hidden, timestep, seqlen)

                if loss is not None:
                    losses_batch.append(loss.data[0])

            losses.append(sum(losses_batch) / len(losses_batch))

        return losses

    @training
    def learn(self, dataset, optimizer, validation_dataset=None, max_epochs=50):
        """
        Optimize the model parameters using the dataset. This function use the algorithm::

            for epoch in max_epochs:
                try:
                    for batch in data:
                        # forward the data
                        hidden = get_initial_state(batch)
                        seqlen = len(batch[0])

                        for idx, timestep in enumerate(batch, 1):
                            output, hidden = forward(timestep, hidden, idx, seqlen)
                            error = loss(output, timestep, hidden, idx, seqlen)

                            if error is not None:
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
        features, it's recommended to use the :class:`~cogitare.data.SequentialDataSet` class. It
        provides a interface that loads batches using multiple threads/processes
        and provides useful tasks such as data splitting, async data loading, shuffling, and more. For
        sequential data with variable length, it can automatically pad the sequences such that all of them
        have the same length.

        Args:
            dataset (iterator): an iterator that returns one batch per iteration.
                Each batch is an iterator, where each item is a sequence. To have a better
                performance and a easy to use interface, it is recommended to
                use the :class:`~cogitare.data.SequentialDataSet`.
            optimizer (torch.optim): the instance of a :class:`torch.optim.Optimizer` object.
            validation_dataset (iterator, optional): if provided, must have the same
                caracteristics that the ``dataset``. This may be used by the model and
                by plugins to evaluate the model performance during training.
            max_epochs (int): the number of epochs before ending the training procedure.

        Returns:
            status (bool): False if stopped by :class:`~cogitare.utils.StopTraining`. True otherwise.
        """
        return super(SequentialModel, self).learn(dataset, optimizer, validation_dataset, max_epochs)
