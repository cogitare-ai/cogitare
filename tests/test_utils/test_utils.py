import unittest
from cogitare import utils, config


class TestUtils(unittest.TestCase):

    def test_verbosity_default(self):
        self.assertEqual(utils.get_verbosity(), config['VERBOSE'])

    def test_verbosity_set(self):
        self.assertEqual(utils.get_verbosity(True), True)
        self.assertEqual(utils.get_verbosity(False), False)

    def test_verbosity_set_boolonly(self):
        self.assertRaises(TypeError, utils.get_verbosity('test'))

    def test_call_feedback_raises(self):
        self.assertRaisesRegexp(ValueError, 'callable nor list', utils.call_feedback, 'test')

        def f():
            return 'test'

        self.assertRaisesRegex(ValueError, 'callable nor list', utils.call_feedback, [f, 'test'])
        self.assertRaisesRegex(ValueError, 'callable nor list',
                               utils.call_feedback, [f, [f, f, 'test']])
