import os
from functools import partial
import psutil
import datetime
import platform
from py3nvml import py3nvml


def _nmvl_call(func):
    try:
        return func()
    except Exception as e:
        return None


def gpu_status():
    try:
        py3nvml.nvmlInit()
        device_count = py3nvml.nvmlDeviceGetCount()

        devices = []
        for i in range(device_count):
            gpu = {}
            handle = py3nvml.nvmlDeviceGetHandleByIndex(i)

            memory = _nmvl_call(partial(py3nvml.nvmlDeviceGetMemoryInfo, handle))
            if memory:
                memory = round(memory.total * 1.0 / 2 ** 30, 2)

            gpu['name'] = _nmvl_call(partial(py3nvml.nvmlDeviceGetName, handle))
            gpu['clock'] = _nmvl_call(partial(
                py3nvml.nvmlDeviceGetApplicationsClock, handle, py3nvml.NVML_CLOCK_GRAPHICS))
            gpu['clock_mem'] = _nmvl_call(partial(
                py3nvml.nvmlDeviceGetApplicationsClock, handle, py3nvml.NVML_CLOCK_MEM))
            gpu['clock_max'] = _nmvl_call(partial(
                py3nvml.nvmlDeviceGetMaxClockInfo, handle, py3nvml.NVML_CLOCK_GRAPHICS))
            gpu['clock_mem_max'] = _nmvl_call(partial(
                py3nvml.nvmlDeviceGetMaxClockInfo, handle, py3nvml.NVML_CLOCK_MEM))
            gpu['memory'] = memory

            devices.append(gpu)
        nvidia = {
            'driver_version': py3nvml.nvmlSystemGetDriverVersion(),
            'devices': devices
        }

        return nvidia
    except Exception as e:
        return None


def machine_status():
    p = psutil.Process()

    stats = {
        'environ': dict(os.environ),
        'pid': os.getpid(),
        'type': platform.machine(),
        'hostname': platform.node(),
        'os': platform.system(),
        'os_release': platform.release(),
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'num_cpu': psutil.cpu_count(),
        'num_cpu_real': psutil.cpu_count(False),
        'usable_cpu': len(p.cpu_affinity()),
        'min_max_cpus': [(a.min, a.max) for a in psutil.cpu_freq(percpu=True)],
        'total_ram': round(1.0 * psutil.virtual_memory().total / 2**30, 2),
        'total_swap': round(1.0 * psutil.swap_memory().total / 2**30, 2),
        'create_time': datetime.datetime.fromtimestamp(p.create_time()).strftime("%Y-%m-%d %H:%M:%S"),
        'nvidia': False
    }

    gpu = gpu_status()
    if gpu is not None:
        stats['nvidia'] = True
        stats['gpu'] = gpu

    return stats
