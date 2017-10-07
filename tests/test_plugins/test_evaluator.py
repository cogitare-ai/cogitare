import unittest
import mock
from cogitare.plugins import Evaluator


class TestEvaluator(unittest.TestCase):

    def test_evaluator(self):
        mock1 = mock.MagicMock(return_value=1)
        mock2 = mock.MagicMock(return_value=2)
        model = mock.Mock()

        data = [1, 2, 3]
        metrics = {'loss': mock1, 'loss2': mock2}
        e = Evaluator(data, metrics)

        e.function(model)

        model.evaluate_with_metrics.assert_called_with(data, metrics)
