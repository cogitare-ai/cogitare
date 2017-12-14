from cogitare import utils
import time
import concurrent.futures as C
from six.moves.queue import PriorityQueue
from threading import Thread


def _identity(x):
    return x


def _fetch(f, data):
    batch = next(data)
    return f(batch)


class AsyncDataLoader(object):
    """The AsyncDataLoader is a wrapper to asynchronous loading multiples batches of data.

    It keeps a buffer of batches, so when the model asks for a new batch, it's
    already in memory. After sending the batch to the model, it is removed from
    the buffer, and a new batch can be loaded.

    The buffer is filled using a separated thread. Then, each batch can be loaded
    using multiple processes, or multiples threads.

    This async batch loader is designed for heavy IO or heavy CPU batch generation.

    .. warning:: When using the multiprocessing batch loader, watch the ram usage,
        and avoid a high number of processes. Multiprocessing can easily lead
        to memory overflow.

    - Should I use the Async Data Loader ?

        If you check the Cogitare's DataHolder, it already provides the execution
        of the data loading through multiple threads or multiple processes. So you should
        use this if the time to generate a whole batch is expensive.

    - Should I use threads or processes ?

        It's recommended to use threads, they are lightweight and fast.

        Multiple processing usually will lead to a worse performance and memory
        usage, due to the communication pipe between processes and due to
        the extension sharing of the memory. However, it can be
        useful for CPU expensive operations, because will not suffer from GIL.

        Threads, in the other way, are lightweight and usually fast, but can suffer from GIL.
        For tasks with heavy IO, it is a good choice.

    Args:
        data (DataSet, AbsDataHolder, SequentialDataSet, SequentialAbsDataHolder): data holder,
            or dataset instance.
        buffer_size (int): size of the batch buffer. The async data loader will keep around
            ``buffer_size`` batches in memory.
        mode (str): should be ``threaded`` or ``multiprocessing``, indicating how to fetch batches.
        workers (int): the number of threads/processes used to load the batches. If None,
            will use the number of cores in the CPU.
        on_batch_loaded (callable): if provided, this function will be called when a new batch is loaded. It must
            receive one argument, the batch data. And return the batch after applying some operation on the data. This
            can be used to apply pre-processing functions on a batch of data (such as image filtering, moving the

    Example::

        >>> mnist = fetch_mldata('MNIST original')
        >>> mnist.data = mnist.data / 255
        >>> data = DataSet([mnist.data, mnist.target.astype(int)], batch_size=64)
        >>> data_train, data_validation = data.split(0.8)

        >>> # wraps the data_train dataset with the async loader.
        >>> data_train = AsyncDataLoader(data_train)

        >>> model.learn(data_train, optimizer)
    """

    def __init__(self, data, buffer_size=8, mode='threaded', workers=None, on_batch_loaded=None):
        valid = ('threaded', 'multiprocessing', )
        utils.assert_raise(mode in valid, ValueError,
                           'mode must be one of: ' + ', '.join(valid))
        utils.assert_raise(buffer_size >= 2, ValueError,
                           'buffer_size must be greater or equal to 2')
        if mode == 'threaded':
            self._executor = C.ThreadPoolExecutor(workers)
        else:
            self._executor = C.ProcessPoolExecutor(workers)

        if on_batch_loaded is None:
            on_batch_loaded = _identity

        self._queue = PriorityQueue(buffer_size)
        self._data = data
        self._thread = None
        self._on_batch_loaded = on_batch_loaded

    def __repr__(self):
        return repr(self._data)

    def _start(self):
        if self._thread is None:
            self._thread = Thread(target=self._produce)
            self._thread.daemon = True
            self._thread.start()

    def cache(self):
        """Start to load batches to buffer, and wait the buffer be full.
        This can be used before start the model training to cache the samples
        and speed up the model execution.

        Example::

            >>> dh = CallableHolder(s.__next__, mode='sequential', total_samples=20000000, single=True)
            >>> dh = AsyncDataLoader(dh, buffer_size=64000, mode='threaded', workers=1)
            >>> print('caching ...')
            >>> dh.cache()
            >>> print('done')
        """

        self._start()
        while not self._queue.full():
            time.sleep(0.1)

    def _produce(self):
        idx = 0

        while True:
            future = self._executor.submit(_fetch, self._on_batch_loaded, self._data)
            self._queue.put((idx, future))
            idx += 1

    def __iter__(self):
        return self

    def __next__(self):
        self._start()
        return self._queue.get()[1].result()

    next = __next__

    def __len__(self):
        return len(self._data)
