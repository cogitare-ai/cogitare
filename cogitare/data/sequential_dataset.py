from cogitare.data.dataset import DataSet
from cogitare.data.sequential_dataholder import SequentialAutoHolder, SequentialAbsDataHolder
from six.moves import zip_longest


class SequentialDataSet(DataSet):
    """This object is a container to multiple sequential data holders. Its an extension of
    :class:`~cogitare.data.DataSet`.

    Check the :class:`cogitare.data.DataSet` docs for more information.

    When creating it, the data parameter must be a list of data sources, and then
    the result of the iterator will be a batch got from each data source that contains
    timesteps for each sample.

    It can be used to group input and target variables, for example, if they are not
    on the same object, or two sequences in the case of a sequence to sequnce (seq2seq),
    encode-decode model.

    Args:
        data (list): list of sequential data holders or list of data (that will be converted to a
            data holder using the :class:`~cogitare.data.SequentialAutoHolder`.
        data_types (dataholder, list): A list of data holder classes to be used to
            create the holder for each input.
            If it is a list, must have the same size of the data.
            If data_types is None, the :class:`~cogitare.data.AutoHolder` will be used.
        batch_size (int): the size of each batch.
        shuffle (bool): if True, shuffles the samples of each data holder.
        drop_last (bool): if True, then skip the batch if its size is lower that **batch_size** (can
            occur in the last batch).
        padding_value: this value will be used to pad sequences with different
            sizes in the same batch. When loading a batch, all sequences will have
            the same size. The padding_value is added to the right of each sequence to match the size
            of the longest sequence in the batch.

    .. note:: The **batch_size**, **shuffle**, and **drop_last** parameters will override the
        data holder default values. If you want to keep a different value for each data holder,
        get its instance in the ``container`` attribute and edit manually.


    Example::

        >>> tensor1 = torch.Tensor([[1,2,3], [4,5,6], [7,8,9]])
        >>> tensor2 = torch.Tensor([[10, 11, 12], [13, 14, 15], [16, 17, 18]])

        >>> tensor1
         1  2  3
         4  5  6
         7  8  9
        [torch.FloatTensor of size 3x3]
        >>> tensor2
         10  11  12
         13  14  15
         16  17  18
        [torch.FloatTensor of size 3x3]

        >>> # you can create the dataset passing the data
        >>> data = SequentialDataSet([tensor1, tensor2], batch_size=2)

        >>> # or passing data holders
        >>> dh1 = SequentialTensorHolder(tensor1)
        >>> dh2 = SequentialTensorHolder(tensor1)
        >>> data = SequentialDataSet([dh1, dh2], batch_size=2)

        >>> batch = next(data)
        >>> batch
        [((4.0, 7.0), (13.0, 16.0)), ((5.0, 8.0), (14.0, 17.0)), ((6.0, 9.0), (15.0, 18.0))]

        >>> for timestep, (seq1, seq2) in enumerate(batch, 1):
        ...   print('Current timestep: ' + str(timestep))
        ...    print((seq1, seq2))
        Current timestep: 1
        ((4.0, 7.0), (13.0, 16.0))
        Current timestep: 2
        ((5.0, 8.0), (14.0, 17.0))
        Current timestep: 3
        ((6.0, 9.0), (15.0, 18.0))
    """

    @property
    def _AutoHolderClass(self):
        return SequentialAutoHolder

    @property
    def _AbsDataHolderClass(self):
        return SequentialAbsDataHolder

    def __init__(self, *args, **kwargs):
        self._padding_value = kwargs.pop('padding_value', None)

        super(SequentialDataSet, self).__init__(*args, **kwargs)

        for c in self.container:
            c._padding_value = self._padding_value

    def __next__(self):
        batch = super(SequentialDataSet, self).__next__()

        return list(zip_longest(*batch, fillvalue=self._padding_value))

    next = __next__
