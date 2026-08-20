from pathlib import Path
import time

PARENT = Path(__file__).parent.parent.name

def clock(fn):
    global PARENT
    def clocked(*args, **kwargs):
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        t1 = time.perf_counter()
        dt = t1 - t0
        print(f"{PARENT}.{fn.__module__}.{fn.__name__}: {dt} ns")
        return result
    return clocked
