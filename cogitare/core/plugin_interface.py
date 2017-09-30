import inspect


class PluginInterface(object):

    @property
    def name(self):
        if self._name:
            return self._name
        return type(self).__name__

    @name.setter
    def name(self, value):
        self._name = value

    @property
    def function(self):
        if self._function is None:
            raise ValueError('function is not defined!')
        return self._function

    @function.setter
    def function(self, value):
        if not callable(value):
            raise ValueError('you must provide a callable object!')
        self._function = value

    def __init__(self, freq=1):
        self.freq = freq
        self._counter = 0
        self._function = None
        self._name = None

    def __call__(self, *args, **kwargs):
        self._counter += 1
        if self._counter % self.freq == 0:
            return self.function(*args, **kwargs)

    def reset(self):
        self._counter = 0

    @classmethod
    def from_function(cls, f, freq=1):
        c = cls(freq)
        c.function = f

        if inspect.ismethod(f) or inspect.isfunction(f):
            c.name = f.__name__
        else:
            c.name = type(f).__name__
        return c
