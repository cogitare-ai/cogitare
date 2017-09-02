from cogitare.data import AutoHolder
from cogitare.data.dataholder import _DataHolder
from cogitare import utils


class DataSet(object):

    def __init__(self, data, data_types=None, batch_size=1, shuffle=True, drop_last=False):
        utils.assert_raise(isinstance(data, list), ValueError,
                           '"data" must be a list of model data')

        self._batch_size = batch_size
        self._shuffle = shuffle
        self._drop_last = drop_last
        if data_types is None:
            data_types = [AutoHolder] * len(data)

        if not isinstance(data_types, list):
            utils.assert_raise(isinstance(data_types, _DataHolder), ValueError,
                               '"data_types" must be a DataHolder class of a list of them')
            data_types = [data_types] * len(data)

        self.container = []
        indices = None
        for i, d in enumerate(data):
            if isinstance(d, _DataHolder):
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
        return """DataSet with:
    containers: [
        {0}
    ],
    batch size: {1}\n""".format('\n\t'.join([repr(c) for c in self.container]),
                                self._batch_size)

    def split(self, ratio):
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
        data = [[] for x in range(n)]

        for i, c in enumerate(self.container):
            folds = c.split_chunks(n)

            for idx, f in enumerate(folds):
                data[idx].append(f)

        return [DataSet(data[i], batch_size=self._batch_size, shuffle=self._shuffle,
                        drop_last=self._drop_last) for i in range(n)]

    def shuffle(self):
        self.container[0].shuffle()

    def reset(self):
        [c.reset() for c in self.container]

    def __len__(self):
        return len(self.container[0])

    def __getitem__(self, key):
        if isinstance(key, slice):
            raise NotImplementedError
        return [c[key] for c in self.container]

    def _get_batch(self):
        return [c._get_batch() for c in self.container]

    def __iter__(self):
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
