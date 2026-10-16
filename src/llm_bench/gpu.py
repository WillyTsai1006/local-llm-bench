import platform
import threading
import time

from pynvml import (
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetMemoryInfo,
    nvmlDeviceGetName,
    nvmlInit,
)


def get_system_info() -> dict:
    nvmlInit()

    handle = nvmlDeviceGetHandleByIndex(0)
    memory = nvmlDeviceGetMemoryInfo(handle)

    return {
        "python_version": platform.python_version(),
        "cuda_available": True,
        "gpu_name": nvmlDeviceGetName(handle),
        "vram_gb": memory.total / (1024**3),
    }

class NvmlMemoryMonitor:
    def __init__(self, interval: float = 0.01):
        nvmlInit()

        self.handle = nvmlDeviceGetHandleByIndex(0)
        self.interval = interval

        self.peak_used_bytes = 0

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample(self) -> None:
        while not self._stop_event.is_set():
            memory = nvmlDeviceGetMemoryInfo(self.handle)

            self.peak_used_bytes = max(
                self.peak_used_bytes,
                memory.used,
            )

            time.sleep(self.interval)

    def start(self) -> None:
        self.peak_used_bytes = 0
        self._stop_event.clear()

        self._thread = threading.Thread(
            target=self._sample,
            daemon=True,
        )

        self._thread.start()

    def stop(self) -> float:
        self._stop_event.set()

        if self._thread is not None:
            self._thread.join()

        memory = nvmlDeviceGetMemoryInfo(self.handle)

        self.peak_used_bytes = max(
            self.peak_used_bytes,
            memory.used,
        )

        return self.peak_used_bytes / (1024**3)