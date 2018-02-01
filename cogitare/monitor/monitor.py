from socketIO_client import SocketIO, BaseNamespace
from threading import Thread
from cogitare.monitor.workers.basic import machine_status
from cogitare.monitor.workers.system_usage import SystemUsage


class CogitareNamespace(BaseNamespace):

    def __init__(self, *args, **kwargs):
        super(CogitareNamespace, self).__init__(*args, **kwargs)
        self._monitor = None
        self._system_usage = SystemUsage(lambda msg: self.emit('update_usage', msg))

    def on_toggle_system_usage(self, *args):
        self._system_usage.enabled = not self._system_usage.enabled


class Monitor(Thread):

    def __init__(self, name, description=None, host='localhost', port='8787', save_on_disk=True):
        super(Monitor, self).__init__(daemon=True)
        self._host = host
        self._port = port
        self._save_on_disk = save_on_disk
        self._description = description
        self._name = name
        self.start()

    def run(self):
        self._client = SocketIO(self._host, self._port, wait_for_connection=False)
        self._namespace = self._client.define(CogitareNamespace, '/cogitare')
        self._namespace._monitor = self
        self._namespace.emit('register', {'name': self._name,
                                          'description': self._description,
                                          'save_on_disk': self._save_on_disk})
        self._namespace.emit('machine', machine_status())
        self._client.wait()
