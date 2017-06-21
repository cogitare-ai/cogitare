import unittest
import mock as m
from cogitare import utils
import pytest


class TestUtils(unittest.TestCase):

    def test_call_feedback_raises(self):
        with pytest.raises(ValueError) as info:
            utils.call_feedback('test')
        self.assertIn('callable nor list', str(info.value))

        def f():
            return 'test'

        with pytest.raises(ValueError) as info:
            utils.call_feedback([f, 'test'])
        self.assertIn('callable nor list', str(info.value))

        with pytest.raises(ValueError) as info:
            utils.call_feedback([f, [f, f, 'test']])
        self.assertIn('callable nor list', str(info.value))

    def test_call_feedback_ignores_empty(self):
        out = utils.call_feedback(None)
        self.assertEqual(out, False)

    def test_call_feedback_call(self):
        def function(*args, **kwargs):
            pass

        f = m.create_autospec(function, return_value=None)
        utils.call_feedback(f)
        f.assert_any_call()

    def test_call_feedback_list_call(self):
        def f1(*args, **kwargs):
            pass

        def f2(*args, **kwargs):
            pass

        def f3(*args, **kwargs):
            pass

        def f4(*args, **kwargs):
            pass

        def f5(*args, **kwargs):
            pass

        fs = [m.create_autospec(f, return_value=None) for f in [f1, f2, f3, f4, f5]]

        feedbacks = [fs[0], fs[1], [fs[2], [fs[3], fs[4]]], fs[0]]
        utils.call_feedback(feedbacks)
        for f in fs:
            f.assert_any_call()
