import time
import functools
from contextlib import contextmanager
from typing import Optional, Dict, List
from collections import defaultdict

class CodeTimer:
    """A class to track execution times of different code components."""
    
    def __init__(self):
        self.timings = defaultdict(list)
        self.total_times = {}
    
    def get_statistics(self, component: str) -> Dict[str, float]:
        """Calculate statistics for a specific component."""
        times = self.timings[component]
        if not times:
            return {}
        
        return {
            'avg_time': sum(times) / len(times),
            'min_time': min(times),
            'max_time': max(times),
            'total_time': sum(times),
            'calls': len(times)
        }
    
    def print_report(self):
        """Print a detailed timing report for all components."""
        print("\n=== Timing Report ===")
        for component in sorted(self.timings.keys()):
            stats = self.get_statistics(component)
            if stats:
                print(f"\nComponent: {component}")
                print(f"  Total calls: {stats['calls']}")
                print(f"  Average time: {stats['avg_time']:.6f} seconds")
                print(f"  Min time: {stats['min_time']:.6f} seconds")
                print(f"  Max time: {stats['max_time']:.6f} seconds")
                print(f"  Total time: {stats['total_time']:.6f} seconds")

# Create a global timer instance
timer = CodeTimer()

@contextmanager
def time_block(name: str):
    """Context manager for timing code blocks."""
    start_time = time.time()
    try:
        yield
    finally:
        end_time = time.time()
        timer.timings[name].append(end_time - start_time)

def time_function(func=None, *, name: Optional[str] = None):
    """Decorator for timing functions."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            component_name = name or func.__name__
            with time_block(component_name):
                return func(*args, **kwargs)
        return wrapper
    
    if func is None:
        return decorator
    return decorator(func)
