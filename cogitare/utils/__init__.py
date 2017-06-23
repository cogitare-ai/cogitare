from torch.nn import Module


_CUDA_ENABLED = False


def not_training(func):
    """decorator do disable the training during execution. Must be used inside a Module class
    """
    def f(self, *args, **kwargs):
        assert isinstance(self, Module)
        default = self.training
        self.train(False)
        func(self, *args, **kwargs)
        self.train(default)

    return f


def training(func):
    """decorator do enable the training during execution. Must be used inside a Module class
    """
    def f(self, *args, **kwargs):
        assert isinstance(self, Module)
        default = self.training
        self.train(True)
        func(self, *args, **kwargs)
        self.train(default)

    return f


def call_feedback(feedback, *args, **kwargs):
    """call the feedback, and if it's iterable, iterates over it and call all instances
    """
    if feedback is None:
        return False
    if callable(feedback):
        return feedback(*args, **kwargs)
    elif isinstance(feedback, (list, tuple)):
        for feed in feedback:
            call_feedback(feed, *args, **kwargs)
    else:
        raise ValueError('feedback is neither callable nor list')


def call_watchdog(watchdog, *args, **kwargs):
    """call the Watchdog, and if it's iterable, iterates over it and call all instances.
    If any of them returns True, this caller will return True, indicating that the model
    should stop the training process.
    """
    if watchdog is None:
        return False
    if callable(watchdog):
        return watchdog(*args, **kwargs)
    elif isinstance(watchdog, (list, tuple)):
        if any([call_watchdog(watcher, *args, **kwargs) is True for watcher in watchdog]):
            return True
    else:
        raise ValueError('watchdog is neither callable nor list')


def assert_raise(valid, exception, msg):
    """if boolean is False, raises a excetion using the provided message
    """
    if not valid:
        raise exception(msg)


def get_cuda(cuda=None):
    """get Cogitare default cuda enabled/disabled."""
    if cuda is None:
        return _CUDA_ENABLED
    return cuda


def set_cuda(cuda):
    """set Cogitare default cuda enabled/disabled."""
    global _CUDA_ENABLED
    _CUDA_ENABLED = cuda
