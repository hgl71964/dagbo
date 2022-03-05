import pynvml


def gpu_usage():
    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(0)
    info = pynvml.nvmlDeviceGetMemoryInfo(h)
    print("gpu usage:")
    print(f'total    : {info.total/ 1024 ** 3:.2f}GB')
    print(f'free     : {info.free/ 1024 ** 3:.2f}GB')
    print(f'used     : {info.used/ 1024 ** 3:.2f}GB')
