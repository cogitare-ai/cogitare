import cogitare.metrics.spatial as S
import math
import torch
from tests.common import TestCase


class TestSpatial(TestCase):

    def test_braycurtis_distance(self):
        v1 = S.braycurtis_distance([1, 2, 3], [2, 4, 6], dim=0)
        self.assertAlmostEqual(float(v1[0]), 1.0 / 3)

        v2 = S.braycurtis_distance([1, 1, 0, 0], [1, 0, 1, 0], dim=0)
        self.assertAlmostEqual(float(v2[0]), 0.5)

        v3 = S.braycurtis_distance([[1, 2, 3], [2, 4, 6]],
                                   [[4, 5, 6], [1, 2, 3]], dim=0)

        v4 = S.braycurtis_distance([[1, 2, 3], [2, 4, 6]],
                                   [[4, 5, 6], [1, 2, 3]], dim=1)

        self.assertEqual(v3, torch.Tensor([0.5, 5.0 / 13, 1.0 / 3]))
        self.assertEqual(v4, torch.Tensor([3.0 / 7, 1.0 / 3]))

    def test_canberra_distance(self):
        v1 = S.canberra_distance([1, 2, 3], [2, 4, 6], dim=0)
        self.assertAlmostEqual(float(v1[0]), 1.0)

        v2 = S.canberra_distance([1, 1, 0, 0], [1, 0, 1, 0], dim=0)
        self.assertAlmostEqual(float(v2[0]), 2.0)

        v3 = S.canberra_distance([[1, 2, 3], [-2, 4, 6]],
                                 [[4, 5, 6], [1, 2, 3]], dim=0)

        v4 = S.canberra_distance([[1, 2, 3], [-2, 4, 6]],
                                 [[4, 5, 6], [1, 2, 3]], dim=1)

        self.assertEqual(v3, torch.Tensor([1.6, 16.0 / 21, 2.0 / 3]))
        self.assertEqual(v4, torch.Tensor([143.0 / 105, 5.0 / 3]))

    def test_cosine(self):
        v1 = S.cosine_distance([1, 2, 3], [1, 2, 3], dim=0)
        v11 = S.cosine_similarity([1, 2, 3], [1, 2, 3], dim=0)
        v2 = S.cosine_distance([1, 2, 3], [1, 2, 2], dim=0)
        v22 = S.cosine_similarity([1, 2, 3], [1, 2, 2], dim=0)
        v3 = S.cosine_distance([[1, 2, 3], [1, 2, 3]],
                               [[1, 2, 3], [1, 2, 2]], dim=0)
        v4 = S.cosine_distance([[1, 2, 3], [1, 2, 3]],
                               [[1, 2, 3], [1, 2, 2]], dim=1)

        self.assertEqual(v1, torch.Tensor([0]))
        self.assertEqual(v11, torch.Tensor([1]))
        self.assertEqual(v2, torch.Tensor([0.020042]))
        self.assertEqual(v2 + v22, torch.Tensor([1]))
        self.assertEqual(v4, torch.Tensor([0, 0.020042]))
        self.assertEqual(v3, torch.Tensor([0, 0, 0.0194193]))

    def test_norm_distance(self):
        v1 = S.norm_distance([1, 2, 3], [1, 3, 5], norm=1, dim=0)
        v2 = S.norm_distance([1, 2, 3], [1, 3, 5], norm=2, dim=0)
        v3 = S.norm_distance([[1, 2, 3], [1, 3, 5]],
                             [[2, 2, 1], [0, 1, 2]], norm=2, dim=0)

        self.assertEqual(v1, torch.Tensor([3]))
        self.assertEqual(v2, torch.Tensor([math.sqrt(5)]))
        self.assertEqual(v3, torch.Tensor([math.sqrt(2), 2, math.sqrt(13)]))

    def test_euclidian_distane(self):
        x1, x2 = [torch.rand(3, 4, 5) for i in range(2)]

        for d in range(3):
            a = S.norm_distance(x1, x2, norm=2)
            b = S.euclidian_distance(x1, x2)
            self.assertEqual(a, b)

    def test_manhattan_distane(self):
        x1, x2 = [torch.rand(3, 4, 5) for i in range(2)]

        for d in range(3):
            a = S.norm_distance(x1, x2, norm=1)
            b = S.manhattan_distance(x1, x2)
            self.assertEqual(a, b)

    def test_chebyshev_distance(self):
        v1 = S.chebyshev_distance([1, 2, 10], [2, 3, 5], dim=0)
        v2 = S.chebyshev_distance([100, 2, 10], [2, 3, 5], dim=0)

        v3 = S.chebyshev_distance([[5, 4, 5], [1, 2, 3]],
                                  [[10, 2, 5], [3, 3, 3]], dim=0)
        v4 = S.chebyshev_distance([[5, 4, 5], [1, 2, 3]],
                                  [[10, 2, 5], [3, 3, 3]], dim=1)

        self.assertEqual(v1, torch.Tensor([5]))
        self.assertEqual(v2, torch.Tensor([98]))
        self.assertEqual(v3, torch.Tensor([5, 2, 0]))
        self.assertEqual(v4, torch.Tensor([5, 2]))
