from threading import Thread
import psutil
import time


class SystemUsage(Thread):

    def __init__(self, callback, *args, **kwargs):
        super(SystemUsage, self).__init__(daemon=True)
        self.interval = 1
        self.enabled = False
        self.callback = callback
        self.p = psutil.Process()

        self.start()

    def run(self):
        while True:
            if self.enabled:
                self.callback(self.get_usage())
            time.sleep(self.interval)

    def get_usage(self):
        usage = {}

        usage['Ram (GB)'] = round(psutil.virtual_memory().used * 1.0 / 2 ** 30, 2)
        usage['CPU (%)'] = dict(('CPU %d' % idx, usage) for idx, usage in enumerate(psutil.cpu_percent(percpu=True), 1))
        usage['CPU used by the Process (%)'] = round(self.p.cpu_percent(), 2)
        usage['RAM used by the Process (MB)'] = round(self.p.memory_info().rss * 1.0 / 2 ** 20, 2)
        usage['Number of Threads'] = self.p.num_threads()

        return usage
