import unittest
import mock as m
from cogitare import utils
import pytest
import torch.nn as nn


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

    def test_call_watchdog_raises(self):
        with pytest.raises(ValueError) as info:
            utils.call_watchdog('test')
        self.assertIn('callable nor list', str(info.value))

        def f():
            return 'test'

        with pytest.raises(ValueError) as info:
            utils.call_watchdog([f, 'test'])
        self.assertIn('callable nor list', str(info.value))

        with pytest.raises(ValueError) as info:
            utils.call_watchdog([f, [f, f, 'test']])
        self.assertIn('callable nor list', str(info.value))

    def test_call_watchdog_ignores_empty(self):
        out = utils.call_watchdog(None)
        self.assertEqual(out, False)

    def test_call_watchdog_call(self):
        def function(*args, **kwargs):
            pass

        f = m.create_autospec(function, return_value=None)
        utils.call_watchdog(f)
        f.assert_any_call()

    def test_call_watchdog_list_call(self):
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

        returns = [False, None, True, False, False]
        fs = [m.create_autospec(f, return_value=returns[i]) for i, f in enumerate([f1, f2, f3, f4, f5])]

        watchdogs = [fs[0], fs[1], [fs[2], [fs[3], fs[4]]], fs[0]]
        resp = utils.call_watchdog(watchdogs)
        for f in fs:
            f.assert_any_call()

        self.assertEqual(resp, True)

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
