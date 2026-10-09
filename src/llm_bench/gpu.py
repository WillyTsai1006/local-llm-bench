import platform

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