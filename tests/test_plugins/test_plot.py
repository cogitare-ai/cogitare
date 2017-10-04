import unittest
import numpy as np
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
        self.p(test=1)
        self.p(test=0)
        self.p(test=-1)

    def test_plot_with_std(self):
        self.p.add_variable('test', 'Test', std_data='test_std')

        self.p(test=1, test_std=[1, 2])
        self.p(test=2, test_std=[1, 2])
        self.p(test=2, test_std=[0, 3])
        self.p(test=2, test_std=[-101])
