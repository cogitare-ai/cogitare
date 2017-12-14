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
        batch_size (int): the size of each batch.
        shuffle (bool): if True, shuffles the samples of each data holder.
        drop_last (bool): if True, then skip the batch if its size is lower that **batch_size** (can
            occur in the last batch).
        total_samples (int, optional): number of samples. If provided and smalled than the maximum number of samples,
            just the subset of size ``total_samples`` will be used.

    .. note:: The **batch_size**, **shuffle**, and **drop_last** parameters will override the
        data holder default values. If you want to keep a different value for each data holder,
        get its instance in the ``container`` attribute and edit manually.
    """

    @property
    def container(self):
        return self._container

    @property
    def _AutoHolderClass(self):
        return AutoHolder

    @property
    def _AbsDataHolderClass(self):
        return AbsDataHolder

    def __init__(self, data, batch_size=1, shuffle=True, drop_last=False, total_samples=None):
        utils.assert_raise(isinstance(data, (list, tuple)), ValueError,
                           '"data" must be a list or a tuple')

        self._batch_size = batch_size
        self._shuffle = shuffle
        self._drop_last = drop_last
        self._total_samples = total_samples

        self._container = self._create_container(data)

        l0 = len(self._container[0])
        for c in self._container[1:]:
            utils.assert_raise(len(c) == l0, ValueError, 'All data must have the same length!')

        indices = self._container[0].indices

        for c in self._container:
            c._indices = indices

        self.reset()

    def _create_container(self, data):
        container = []
        for d in data:
            if isinstance(d, self._AbsDataHolderClass):
                d._batch_size = self._batch_size
                d._shuffle = self._shuffle
                d._drop_last = self._drop_last
                if self._total_samples is not None:
                    d.total_samples = self._total_samples
            else:
                d = self._AutoHolderClass(d, batch_size=self._batch_size, shuffle=self._shuffle,
                                          drop_last=self._drop_last, total_samples=self._total_samples)

            container.append(d)

        return container

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
            ratio (:obj:`float`): ratio of the split. Must be between 0 and 1.

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

        data1 = self.__class__(d1, batch_size=self._batch_size, shuffle=self._shuffle,
                               drop_last=self._drop_last)
        data2 = self.__class__(d2, batch_size=self._batch_size, shuffle=self._shuffle,
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

        return [self.__class__(data[i], batch_size=self._batch_size, shuffle=self._shuffle,
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
        return [c[key] for c in self.container]

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
