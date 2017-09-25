from cogitare.data.dataset import DataSet
from cogitare.data.sequential_dataholder import SequentialAutoHolder, SequentialAbsDataHolder
from six.moves import zip_longest


class SequentialDataSet(DataSet):

    _AutoHolderClass = SequentialAutoHolder
    _AbsDataHolderClass = SequentialAbsDataHolder

    def __init__(self, *args, **kwargs):
        self._padding_value = kwargs.pop('padding_value', None)

        super(SequentialDataSet, self).__init__(*args, **kwargs)

        for c in self.container:
            c._padding_value = self._padding_value

    def __next__(self):
        batch = super(SequentialDataSet, self).__next__()

        return list(zip_longest(*batch, fillvalue=self._padding_value))

    next = __next__
