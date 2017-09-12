from cogitare.data import AutoHolder
from cogitare.data.dataholder import AbsDataHolder
from cogitare import utils


class DataSet(object):
    """This object is a container to multiple data holders.

    When creating it, the data parameter must be a list of data sources, and then
    the result of the iterator will be a batch got from each data source.

    It can be used to group input and target variables, for example, if they are not
    on the same object.

    Args:
        data (list): list of data holders or list of data (that will be converted to a
            data holder using the :class:`~cogitare.data.AutoHolder`.
        data_types (dataholder, list): A list of data holder classes to be used to
            create the holder for each input.
            If it is a list, must have the same size of the data.
            If data_types is None, the :class:`~cogitare.data.AutoHolder` will be used.
        batch_size (int): the size of each batch.
        shuffle (bool): if True, shuffles the samples of each data holder.
        drop_last (bool): if True, then skip the batch if its size is lower that **batch_size** (can
            occur in the last batch).

    .. note:: The **batch_size**, **shuffle**, and **drop_last** parameters will override the
        data holder default values.
    """

    def __init__(self, data, data_types=None, batch_size=1, shuffle=True, drop_last=False):
        utils.assert_raise(isinstance(data, list), ValueError,
                           '"data" must be a list of model data')

        self._batch_size = batch_size
        self._shuffle = shuffle
        self._drop_last = drop_last
        if data_types is None:
            data_types = [AutoHolder] * len(data)

        if not isinstance(data_types, list):
            utils.assert_raise(isinstance(data_types, AbsDataHolder), ValueError,
                               '"data_types" must be a DataHolder class of a list of them')
            data_types = [data_types] * len(data)

        self.container = []
        indices = None
        for i, d in enumerate(data):
            if isinstance(d, AbsDataHolder):
                d._batch_size = batch_size
                d._shuffle = shuffle
                d._drop_last = drop_last
            else:
                d = data_types[i](d, batch_size=batch_size, shuffle=shuffle,
                                  drop_last=drop_last)

            self.container.append(d)

        l0 = len(self.container[0])
        for c in self.container[1:]:
            utils.assert_raise(len(c) == l0, ValueError, 'All data must have the same length!')

        indices = list(range(self.container[0].total_samples))

        for c in self.container:
            c._indices = indices

        self.reset()

    def __repr__(self):
        """Display a summary of the dataset when using str(dataset) or repr(dataset).
        """
        return """DataSet with:
    containers: [
        {0}
    ],
    batch size: {1}\n""".format('\n\t'.join([repr(c) for c in self.container]),
                                self._batch_size)

    def split(self, ratio):
        """Check :meth:`cogitare.data.AbsDataHolder.split`

        Split the :class:`~cogitare.data.DataSet` into two :class:`~cogitare.data.DataSet`.

        Args:
            ratio (float): ratio of the split. Must be between 0 and 1.

        Returns:
            (data1, data2): two :class:`~cogitare.data.DataSet`.

        Example::

            >>> print(dataset)
            DataSet with:
                containers: [
                    TensorHolder with 1094x64 samples
                    TensorHolder with 1094x64 samples
                ],
                batch size: 64

            >>> ds1, ds2 = data.split(0.8)
            >>> print(ds1)
            DataSet with:
                containers: [
                    TensorHolder with 875x64 samples
                    TensorHolder with 875x64 samples
                ],
                batch size: 64

            >>> print(ds2)
            DataSet with:
                containers: [
                    TensorHolder with 219x64 samples
                    TensorHolder with 219x64 samples
                ],
                batch size: 64
        """
        utils.assert_raise(0 < ratio < 1, ValueError, '"ratio" must be between 0 and 1')

        d1, d2 = [], []

        for c in self.container:
            a, b = c.split(ratio)
            d1.append(a)
            d2.append(b)

        data1 = DataSet(d1, batch_size=self._batch_size, shuffle=self._shuffle,
                        drop_last=self._drop_last)
        data2 = DataSet(d2, batch_size=self._batch_size, shuffle=self._shuffle,
                        drop_last=self._drop_last)

        return data1, data2

    def split_chunks(self, n):
        """Check :meth:`cogitare.data.AbsDataHolder.split_chunks`

        Split the :class:`~cogitare.data.DataSet` into N :class:`~cogitare.data.DataSet`
        with the sample number of samples each.

        Args:
            n (int): number of new splits.

        Returns:
            output (list): list of N :class:`~cogitare.data.DataSet`.

        Example::

            >>> print(dataset)
            DataSet with:
                containers: [
                    TensorHolder with 1094x64 samples
                    TensorHolder with 1094x64 samples
                ],
                batch size: 64

            >>> ds1, ds2, ds3 = dataset.split_chunks(3)
            >>> print(data1)
            DataSet with:
                containers: [
                    TensorHolder with 365x64 samples
                    TensorHolder with 365x64 samples
                ],
                batch size: 64

            >>> print(data2)
            DataSet with:
                containers: [
                    TensorHolder with 365x64 samples
                    TensorHolder with 365x64 samples
                ],
                batch size: 64

            >>> print(data3)
            DataSet with:
                containers: [
                    TensorHolder with 365x64 samples
                    TensorHolder with 365x64 samples
                ],
                batch size: 64

        """
        data = [[] for x in range(n)]

        for i, c in enumerate(self.container):
            folds = c.split_chunks(n)

            for idx, f in enumerate(folds):
                data[idx].append(f)

        return [DataSet(data[i], batch_size=self._batch_size, shuffle=self._shuffle,
                        drop_last=self._drop_last) for i in range(n)]

    def shuffle(self):
        """Shuffles all data holders in this container.

        .. note:: The shuffle keeps all data holder indices aligned. So if you have
            two data holders, one for x data and the other one for y data, for example,
            the pair of samples (x, y) will be shuffled together.
        """
        self.container[0].shuffle()

    def reset(self):
        """Check :meth:`cogitare.data.AbsDataHolder.reset`
        This reset all dataholder's iterators.
        """
        [c.reset() for c in self.container]

    def __len__(self):
        """Returns the number of batches in the dataset
        """
        return len(self.container[0])

    def __getitem__(self, key):
        """Returns a list of samples, getting the key-th sample for each data holder
        in the dataset.
        """
        if isinstance(key, slice):
            raise NotImplementedError
        return [c[key] for c in self.container]

    def _get_batch(self):
        return [c._get_batch() for c in self.container]

    def __iter__(self):
        """Creates an iterator to iterate over batches in the dataset.

        After each iteration over the batches, the dataset will be shuffled if
        the **shuffle** parameter is True.

        Each element of the iterator is a list with the length of data holders. Each
        item of the list is the n-th batch of each data holder.

        Example::

            dataset = DataSet([dataholder1, dataholder2])
            for data1, data2 in dataset:
                print(data1, data2)
        """
        return self

    def __next__(self):
        result = []
        exit = False
        for c in self.container:
            try:
                result.append(next(c))
            except StopIteration:
                exit = True

        if exit:
            raise StopIteration

        return result

    next = __next__
