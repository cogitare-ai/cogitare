from tests.common import TestCase
import pytest
import torch
import cogitare.metrics.classification as C


class TestAccuracy(TestCase):

    def test_1d(self):
        c1 = C.accuracy([1, 2, 3], [4, 5, 6])
        c2 = C.accuracy([1, 1, 1], [1, 1, 1])
        c3 = C.accuracy([1, 2], [1, 1])
        c4 = C.accuracy(torch.LongTensor([1, 2, 3, 4]), torch.FloatTensor([1, 2, 2, 2]))

        self.assertEqual(c1, torch.Tensor([0]))
        self.assertEqual(c2, torch.Tensor([1]))
        self.assertEqual(c3, torch.Tensor([0.5]))
        self.assertEqual(c4, torch.Tensor([0.5]))

    def test_2d(self):
        c1 = C.accuracy([[1, 2], [3, 4], [5, 6]], [[1, 1], [2, 3], [5, 6]])
        c2 = C.accuracy(torch.Tensor([[1, 2, 3], [3, 4, 5]]),
                        torch.Tensor([[1, 2, 3], [1, 2, 3]]))

        self.assertEqual(c1, torch.Tensor([0.5, 0, 1]))
        self.assertEqual(c2, torch.Tensor([1, 0]))

    def test_wrong_dim(self):
        a = torch.Tensor(2, 3)
        b = torch.Tensor(3, 3, 3)

        with pytest.raises(ValueError) as info:
            C.accuracy(a, b)
        self.assertIn('on "expected". Got', str(info.value))

        with pytest.raises(ValueError) as info:
            C.accuracy(b, a)
        self.assertIn('on "prediction". Got', str(info.value))

    def test_diff_size(self):
        a = torch.Tensor(10)
        b = torch.Tensor(1, 10)

        with pytest.raises(ValueError) as info:
            C.accuracy(a, b)
        self.assertIn('must have the same dimension.', str(info.value))
