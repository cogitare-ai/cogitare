import unittest
from cogitare.utils import StopTraining
import pytest
import mock
from cogitare.plugins import EarlyStopping


class TestEarlyStopping(unittest.TestCase):

    def test_create(self):
        EarlyStopping(max_tries=5, path='/tmp/tst')

    def setUp(self):
        model = mock.Mock()
        model.save = mock.MagicMock(return_value=None)
        model.load = mock.MagicMock(return_value=None)

        self.model = model

    def test_save(self):
        e = EarlyStopping(10, '/tmp/test')
        e.function(self.model, 1, 1, 1, 1)

        assert self.model.save.called
        assert not self.model.load.called

        for i in range(2, 11):
            e.function(self.model, i, i)

        self.assertEqual(self.model.save.call_count, 1)
        self.assertEqual(self.model.load.call_count, 0)

        e.function(self.model, 0.5, 11)

        for i in range(11, 22):
            e.function(self.model, i, i)

        self.assertEqual(self.model.save.call_count, 2)
        self.assertEqual(self.model.load.call_count, 0)

        with pytest.raises(StopTraining):
            e.function(self.model, 1, 22)

        self.assertEqual(self.model.save.call_count, 2)
        self.assertEqual(self.model.load.call_count, 1)
