import torch
import numpy as np
import mock as m
from cogitare import utils
import pytest
import torch.nn as nn
from tests.common import TestCase


class TestUtils(TestCase):

    def test_assert_raise(self):
        for exp in (ValueError, IndexError, Exception):
            with pytest.raises(exp) as info:
                utils.assert_raise(1 == 2, exp, 'test message')
            self.assertIn('test message', str(info.value))
        utils.assert_raise(1 == 1, ValueError, 'not raises')

    def test_get_cuda(self):
        self.assertEqual(utils.get_cuda(), utils._CUDA_ENABLED)
        self.assertEqual(utils.get_cuda(True), True)
        self.assertEqual(utils.get_cuda(False), False)
        self.assertEqual(utils.get_cuda(None), utils._CUDA_ENABLED)

    @m.patch('cogitare.utils._CUDA_ENABLED', True)
    def test_set_cuda(self):
        self.assertEqual(utils._CUDA_ENABLED, True)
        utils.set_cuda(False)
        self.assertEqual(utils._CUDA_ENABLED, False)
        utils.set_cuda(True)
        self.assertEqual(utils._CUDA_ENABLED, True)

    def test_training(self):
        def test_training(instance, expected):
            self.assertEqual(instance.training, expected)

        class TestRun(nn.Module):

            @utils.training
            def run_true(self):
                test_training(self, True)

            @utils.not_training
            def run_false(self):
                test_training(self, False)

        t = TestRun()
        t.train(False)
        t.run_true()
        self.assertEqual(t.training, False)

        t.train(True)
        t.run_false()
        self.assertEqual(t.training, True)

    def test_to_tensor(self):
        expected = torch.LongTensor([[1, 2, 3], [4, 5, 6]])
        expected_float = torch.DoubleTensor([[1, 2, 3], [4, 5, 6]])

        from_external = [
            expected,
            [[1, 2, 3], [4, 5, 6]],
            np.asarray([[1, 2, 3], [4, 5, 6]]),
            [np.asarray([1, 2, 3]), np.asarray([4, 5, 6])],
        ]

        for external in from_external:
            self.assertEqual(utils.to_tensor(external), expected)

        for external in from_external:
            self.assertEqual(utils.to_tensor(external, torch.DoubleTensor), expected_float)

        with pytest.raises(ValueError) as info:
            utils.to_tensor({})
        self.assertIn('Invalid data type: dict', str(info.value))

        with pytest.raises(ValueError) as info:
            utils.to_tensor([tuple()])
        self.assertIn('Invalid data type: tuple', str(info.value))

        with pytest.raises(ValueError) as info:
            utils.to_tensor([])
        self.assertIn('Empty list', str(info.value))
