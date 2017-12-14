from tests.common import TestCase
import pytest
import torch
from cogitare.data import TensorHolder, AsyncDataLoader


class TestAsyncLoader(TestCase):

    def setUp(self):
        self.data = torch.rand(100, 32)
        self.dh = TensorHolder(self.data, batch_size=5)
        self.loader = AsyncDataLoader(self.dh, buffer_size=13)

    def test_check_mode(self):
        AsyncDataLoader(self.dh, mode='threaded')
        AsyncDataLoader(self.dh, mode='multiprocessing')

        with pytest.raises(ValueError) as info:
            AsyncDataLoader(self.dh, mode='sequential')
        self.assertIn('mode must be one of:', str(info.value))

        self.assertEqual(len(self.loader), 20)

    def test_cache(self):
        self.assertEqual(self.loader._queue.qsize(), 0)
        self.loader.cache()
        self.assertEqual(self.loader._queue.qsize(), 13)

    def test_iter(self):
        for mode in ['threaded', 'multiprocessing']:
            loader = AsyncDataLoader(self.dh, buffer_size=13, mode=mode)
            loader.cache()
            for batch in self.loader:
                self.assertEqual(len(batch), 5)
            loader.cache()

    def test_repr(self):
        self.assertEqual(repr(self.dh), repr(self.loader))
