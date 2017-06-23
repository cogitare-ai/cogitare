import mock
import cogitare
import unittest


class TestCogitare(unittest.TestCase):

    @mock.patch('torch.manual_seed')
    @mock.patch('torch.cuda.manual_seed')
    def test_seed(self, mock_th, mock_thc):
        cogitare.seed(333)
        mock_th.assert_called_with(333)
        mock_thc.assert_called_with(333)
