from abc import ABCMeta, abstractmethod
from six import add_metaclass


@add_metaclass(ABCMeta)
class Feedback:

    @abstractmethod
    def update(self, instance=None, idx=None, input=None, output=None, target=None, loss=None):
        pass
