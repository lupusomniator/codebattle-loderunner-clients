import time
from collections import defaultdict, Counter
import traceback 

class PerfStat:
    PERF_SUM_VALUE_CACHE = defaultdict(int)
    PERF_COUNT_GLOBAL_CACHE = Counter()
    PERF_LAST_GLOBAL_CACHE = Counter()

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            traceback.print_exception(exc_type, exc_val, exc_tb)
            return False
        elapsed = time.time() - self.start

        PerfStat.PERF_SUM_VALUE_CACHE[self.name] += elapsed
        PerfStat.PERF_COUNT_GLOBAL_CACHE[self.name] += 1
        PerfStat.PERF_LAST_GLOBAL_CACHE[self.name] = elapsed

    @staticmethod
    def print_stat():
        for key in PerfStat.PERF_SUM_VALUE_CACHE:
            print(key)
            print("    total=", PerfStat.PERF_SUM_VALUE_CACHE[key])
            print("    mean=", PerfStat.PERF_SUM_VALUE_CACHE[key] / PerfStat.PERF_COUNT_GLOBAL_CACHE[key])
            print("    last=", PerfStat.PERF_LAST_GLOBAL_CACHE[key])
            

def count_perf(func):
    def wrapper(*args, **kwargs):
        with PerfStat(func.__name__):
            return func(*args, **kwargs)
    return wrapper