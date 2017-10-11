import torch
from mock import patch
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
            [torch.Tensor([1, 2, 3]), torch.Tensor([4, 5, 6])]
        ]

        for external in from_external:
            self.assertEqual(utils.to_tensor(external), expected)

        for external in from_external:
            self.assertEqual(utils.to_tensor(external, torch.DoubleTensor), expected_float)

        data = [[1.0, 2, 3], [4, 5, 6]]
        self.assertEqual(utils.to_tensor(data), expected_float)

        with pytest.raises(ValueError) as info:
            utils.to_tensor({})
        self.assertIn('Invalid data type: dict', str(info.value))

        with pytest.raises(ValueError) as info:
            utils.to_tensor(['asd'])
        self.assertIn('Invalid data type: str', str(info.value))

        with pytest.raises(ValueError) as info:
            utils.to_tensor([])
        self.assertIn('Empty list', str(info.value))

        with pytest.raises(ValueError) as info:
            utils.to_tensor([[torch.Tensor([1, 2, 3]), torch.Tensor([4, 5, 6])],
                             [torch.Tensor([7, 8, 9]), torch.Tensor([10, 11, 12])]])
        self.assertIn('Cannot convert nested list of tensors', str(info.value))

    def test_to_cuda(self):
        tensor = torch.Tensor([1, 2, 3])

        with patch.object(tensor, 'cuda', return_value=None) as mock_method:
            utils.to_tensor(tensor, use_cuda=True)
        mock_method.assert_any_call()

    def test_tensorfy(self):

        @utils.tensorfy(0, 1, tensor_klass=torch.LongTensor)
        def f1(a, b):
            self.assertIsInstance(a, torch.LongTensor)
            self.assertIsInstance(b, torch.LongTensor)

        @utils.tensorfy(0, 1, tensor_klass=torch.DoubleTensor)
        def f2(a, b, c=None):
            self.assertIsInstance(a, torch.DoubleTensor)
            self.assertIsInstance(b, torch.DoubleTensor)
            if c is not None:
                self.assertIsInstance(b, torch.DoubleTensor)

        f1([1, 2, 3], [4, 5, 6])
        f1([[1, 2], [3, 4]], [1, 2, 3, 4])

        f2([1, 2, 3], [4, 5, 6])
        f2([[1, 2], [3, 4]], [1, 2, 3, 4])
        f2([1, 2, 3], [4, 5, 6], [4, 5, 5, 5])
        f2([1, 2, 3], [4, 5, 6], c=[4, 5, 5, 5])

    def test_assert_dim(self):
        with pytest.raises(ValueError) as info:
            utils.assert_dim(torch.rand(3, 3), 'test_tensor', [3])
        self.assertIn('Expected 3D tensor on "test_tensor". Got 2D tensor instead', str(info.value))

        with pytest.raises(ValueError) as info:
            utils.assert_dim(torch.rand(3, 4, 5), 'test', [4, 5, 6])
        self.assertIn('Expected 4D/5D/6D tensor on "test". Got 3D tensor instead', str(info.value))
