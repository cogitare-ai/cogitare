from cogitare import utils
import concurrent.futures as C
from six.moves.queue import PriorityQueue
from threading import Thread


class AsyncDataLoader(object):

    def __init__(self, data, buffer_size=8, mode='threaded', workers=None):
        valid = ('threaded', 'multiprocessing', )
        utils.assert_raise(mode in valid, ValueError,
                           'mode must be one of: ' + ', '.join(valid))
        utils.assert_raise(buffer_size >= 2, ValueError,
                           'buffer_size must be greater or equal to 2')
        if mode == 'threaded':
            self._executor = C.ThreadPoolExecutor(workers)
        else:
            self._executor = C.ProcessPoolExecutor(workers)

        self._queue = PriorityQueue(buffer_size)
        self._data = data
        self._thread = None

    def _start(self):
        self._thread = Thread(target=self._produce, daemon=True)
        self._thread.start()

    def _produce(self):
        idx = 0
        while True:
            future = self._executor.submit(next, self._data)
            self._queue.put((idx, future))
            idx += 1

    def __iter__(self):
        if self._thread is None:
            self._start()
        return self

    def __next__(self):
        return self._queue.get()[1].result()

    next = __next__

    def __len__(self):
        return len(self._data)
