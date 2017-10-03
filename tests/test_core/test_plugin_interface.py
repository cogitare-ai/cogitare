import unittest
import pytest
from cogitare.core import PluginInterface


class TestPluginInterface(unittest.TestCase):

    def test_from_function(self):
        def f1(*args):
            return 1

        def f2(*args):
            return 2

        class F3(object):
            def __call__(self):
                return 3

            def f4(self):
                return 4

        f3 = F3()

        plugin1 = PluginInterface.from_function(f1)
        plugin2 = PluginInterface.from_function(f2, 2)
        plugin3 = PluginInterface.from_function(f3, freq=3)
        plugin4 = PluginInterface.from_function(f3.f4)

        self.assertEqual(plugin1.name, 'f1')
        self.assertEqual(plugin2.name, 'f2')
        self.assertEqual(plugin3.name, 'F3')
        self.assertEqual(plugin4.name, 'f4')

        self.assertEqual(1, plugin1())
        self.assertEqual(4, plugin4())

        self.assertEqual(None, plugin2())
        self.assertEqual(2, plugin2())

        self.assertEqual(None, plugin3())
        self.assertEqual(None, plugin3())
        self.assertEqual(3, plugin3())
        self.assertEqual(None, plugin3())
        self.assertEqual(None, plugin3())
        plugin3.reset()
        self.assertEqual(None, plugin3())
        self.assertEqual(None, plugin3())
        self.assertEqual(3, plugin3())

    def test_name(self):
        def f1(*args):
            pass

        class Plugin(PluginInterface):
            pass

        p = Plugin()
        p1 = PluginInterface.from_function(f1)

        self.assertEqual('f1', p1.name)
        self.assertEqual('Plugin', p.name)
        p.name = 'test'
        self.assertEqual('test', p.name)

    def test_function(self):
        p = PluginInterface()

        def f1():
            return 1

        with pytest.raises(ValueError) as info:
            p.function

        self.assertIn('function is not defined', str(info.value))
        p.function = f1
        self.assertEqual(1, p())

        with pytest.raises(ValueError) as info:
            p.function = 'test'

        self.assertIn('you must provide a callable', str(info.value))
