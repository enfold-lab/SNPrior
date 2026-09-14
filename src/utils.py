import time
import logging
from datetime import datetime
from pathlib import Path
from functools import wraps

from memory_profiler import memory_usage


def configure_logging(log_dir):
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f"run_{datetime.now():%Y%m%d_%H%M%S}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
        force=True,
    )

    return log_file


def measure_performance(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_wall = time.time()
        start_cpu = time.process_time()
        result = memory_usage((func, args, kwargs), max_usage=True, retval=True) #interval=0.1, timeout=None
        end_cpu = time.process_time()
        end_wall = time.time()
        
        cpu_time = end_cpu - start_cpu  # CPU time in seconds
        wall_time = end_wall - start_wall  # Physical time in seconds
        memory_usage_val = result[0] / 1024  # Convert MiB to GiB

        performance_data = {
            # 'function_name': func.__name__,
            'cpu_time': cpu_time / 60,  # Convert seconds to minutes
            'wall_time': wall_time / 60,  # Convert seconds to minutes
            'memory_usage': memory_usage_val  # Memory usage in GiB
        }
        
        return result[1], performance_data  # Return the function's original return value and performance data
    return wrapper

