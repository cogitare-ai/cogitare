import unittest
import pytest
from cogitare.plugins import PlottingMatplotlib


class TestPlottingMatplotlib(unittest.TestCase):

    def setUp(self):
        self.p = PlottingMatplotlib()

    def test_plot(self):
        self.p.add_variable('test', 'Test')

        with pytest.raises(KeyError) as info:
            self.p()
        self.assertIn('test', str(info.value))

        self.p(test=1)
        self.p(test=2)
        self.p(test=3)
        self.p(test=2)
        self.p(test=[1, 2, 3, 4, 5])
        self.p(test=0)
        self.p(test=-1)

    def test_plot_with_std(self):
        self.p.add_variable('test', 'Test', use_std=True)

        self.p(test=1)
        self.p(test=[1, 2, 3, 4])
        self.p(test=2)
        self.p(test=2)
